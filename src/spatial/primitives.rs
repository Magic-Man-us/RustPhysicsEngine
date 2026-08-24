//! Geometric primitive types shared by the intersection, distance,
//! containment, and acceleration modules.
//!
//! Conventions: plane as n·p + d = 0 with unit normal; ray directions
//! normalized by the constructor; polygons CCW-positive.

use crate::linalg::Mat3;
use crate::math::{Vec2, Vec3};
use crate::spatial::mat4::Mat4;

/// Ray with normalized direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    /// # Panics
    /// Panics on a zero direction (which cannot be normalized).
    #[must_use]
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        let d = dir.normalized();
        assert!(d.magnitude_squared() > 0.0, "Ray requires a non-zero direction");
        Self { origin, dir: d }
    }

    /// Point at parameter t: origin + t·dir.
    #[must_use]
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// 3-D line segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub a: Vec3,
    pub b: Vec3,
}

/// 2-D line segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment2 {
    pub a: Vec2,
    pub b: Vec2,
}

/// Plane n·p + d = 0 with unit normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f64,
}

impl Plane {
    /// Plane through a point with the given (normalized) normal.
    ///
    /// # Panics
    /// Panics on a zero normal.
    #[must_use]
    pub fn from_point_normal(p: Vec3, normal: Vec3) -> Self {
        let n = normal.normalized();
        assert!(n.magnitude_squared() > 0.0, "Plane requires a non-zero normal");
        Self { normal: n, d: -n.dot(&p) }
    }

    /// Plane through three points (CCW normal); `None` when collinear.
    #[must_use]
    pub fn from_three_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let n = (b - a).cross(&(c - a));
        if n.magnitude_squared() < 1e-24 {
            return None;
        }
        Some(Self::from_point_normal(a, n))
    }

    /// Signed distance: positive on the normal side.
    #[must_use]
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        self.normal.dot(&p) + self.d
    }

    /// Orthogonal projection onto the plane.
    #[must_use]
    pub fn project(&self, p: Vec3) -> Vec3 {
        p - self.normal * self.signed_distance(p)
    }

    /// The same plane with the opposite orientation.
    #[must_use]
    pub fn flip(&self) -> Self {
        Self { normal: -self.normal, d: -self.d }
    }
}

/// Sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

/// Circle in the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f64,
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Smallest box containing all points.
    ///
    /// # Panics
    /// Panics on an empty slice.
    #[must_use]
    pub fn from_points(points: &[Vec3]) -> Self {
        assert!(!points.is_empty(), "Aabb::from_points requires points");
        let mut min = points[0];
        let mut max = points[0];
        for p in &points[1..] {
            min = Vec3::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Vec3::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }
        Self { min, max }
    }

    /// Smallest box containing both.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Overlapping region, `None` when disjoint.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let min = Vec3::new(
            self.min.x.max(other.min.x),
            self.min.y.max(other.min.y),
            self.min.z.max(other.min.z),
        );
        let max = Vec3::new(
            self.max.x.min(other.max.x),
            self.max.y.min(other.max.y),
            self.max.z.min(other.max.z),
        );
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Center point.
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-widths per axis.
    #[must_use]
    pub fn extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Total surface area.
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Volume.
    #[must_use]
    pub fn volume(&self) -> f64 {
        let d = self.max - self.min;
        d.x * d.y * d.z
    }

    /// Closed containment test.
    #[must_use]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Box grown by `margin` on every side.
    #[must_use]
    pub fn expand(&self, margin: f64) -> Self {
        let m = Vec3::new(margin, margin, margin);
        Self { min: self.min - m, max: self.max + m }
    }

    /// The eight corner points.
    #[must_use]
    pub fn corners(&self) -> [Vec3; 8] {
        let (lo, hi) = (self.min, self.max);
        [
            Vec3::new(lo.x, lo.y, lo.z),
            Vec3::new(hi.x, lo.y, lo.z),
            Vec3::new(lo.x, hi.y, lo.z),
            Vec3::new(hi.x, hi.y, lo.z),
            Vec3::new(lo.x, lo.y, hi.z),
            Vec3::new(hi.x, lo.y, hi.z),
            Vec3::new(lo.x, hi.y, hi.z),
            Vec3::new(hi.x, hi.y, hi.z),
        ]
    }

    /// Axis-aligned bounds of the transformed corners.
    #[must_use]
    pub fn transform(&self, m: &Mat4) -> Aabb {
        let pts: Vec<Vec3> = self.corners().iter().map(|&c| m.transform_point(c)).collect();
        Aabb::from_points(&pts)
    }
}

/// Axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    /// Smallest rectangle containing all points.
    ///
    /// # Panics
    /// Panics on an empty slice.
    #[must_use]
    pub fn from_points(points: &[Vec2]) -> Self {
        assert!(!points.is_empty(), "Rect::from_points requires points");
        let mut min = points[0];
        let mut max = points[0];
        for p in &points[1..] {
            min = Vec2::new(min.x.min(p.x), min.y.min(p.y));
            max = Vec2::new(max.x.max(p.x), max.y.max(p.y));
        }
        Self { min, max }
    }

    /// Smallest rectangle containing both.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vec2::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Vec2::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }

    /// Overlapping region, `None` when disjoint.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let min = Vec2::new(self.min.x.max(other.min.x), self.min.y.max(other.min.y));
        let max = Vec2::new(self.max.x.min(other.max.x), self.max.y.min(other.max.y));
        if min.x <= max.x && min.y <= max.y {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Center point.
    #[must_use]
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Half-widths per axis.
    #[must_use]
    pub fn extents(&self) -> Vec2 {
        (self.max - self.min) * 0.5
    }

    /// Area.
    #[must_use]
    pub fn area(&self) -> f64 {
        let d = self.max - self.min;
        d.x * d.y
    }

    /// Perimeter length.
    #[must_use]
    pub fn perimeter(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x + d.y)
    }

    /// Closed containment test.
    #[must_use]
    pub fn contains_point(&self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Rectangle grown by `margin` on every side.
    #[must_use]
    pub fn expand(&self, margin: f64) -> Self {
        let m = Vec2::new(margin, margin);
        Self { min: self.min - m, max: self.max + m }
    }

    /// The four corner points (CCW from min).
    #[must_use]
    pub fn corners(&self) -> [Vec2; 4] {
        [
            self.min,
            Vec2::new(self.max.x, self.min.y),
            self.max,
            Vec2::new(self.min.x, self.max.y),
        ]
    }
}

/// Oriented bounding box: rotation columns are the local axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obb {
    pub center: Vec3,
    pub half_extents: Vec3,
    pub rotation: Mat3,
}

impl Obb {
    /// Local axes (columns of the rotation).
    #[must_use]
    pub fn axes(&self) -> [Vec3; 3] {
        let d = &self.rotation.data;
        [
            Vec3::new(d[0][0], d[1][0], d[2][0]),
            Vec3::new(d[0][1], d[1][1], d[2][1]),
            Vec3::new(d[0][2], d[1][2], d[2][2]),
        ]
    }

    /// The eight corner points.
    #[must_use]
    pub fn corners(&self) -> [Vec3; 8] {
        let [ax, ay, az] = self.axes();
        let h = self.half_extents;
        let mut out = [Vec3::ZERO; 8];
        for (i, c) in out.iter_mut().enumerate() {
            let sx = if i & 1 == 0 { -1.0 } else { 1.0 };
            let sy = if i & 2 == 0 { -1.0 } else { 1.0 };
            let sz = if i & 4 == 0 { -1.0 } else { 1.0 };
            *c = self.center + ax * (sx * h.x) + ay * (sy * h.y) + az * (sz * h.z);
        }
        out
    }

    /// Tight axis-aligned bounds: center ± Σ |axisᵢ|·hᵢ.
    #[must_use]
    pub fn to_aabb(&self) -> Aabb {
        let [ax, ay, az] = self.axes();
        let h = self.half_extents;
        let e = Vec3::new(
            ax.x.abs() * h.x + ay.x.abs() * h.y + az.x.abs() * h.z,
            ax.y.abs() * h.x + ay.y.abs() * h.y + az.y.abs() * h.z,
            ax.z.abs() * h.x + ay.z.abs() * h.y + az.z.abs() * h.z,
        );
        Aabb { min: self.center - e, max: self.center + e }
    }

    /// PCA-fitted box: axes from the principal directions of the point
    /// covariance (local 3×3 Jacobi), extents from the projections.
    ///
    /// # Panics
    /// Panics on an empty slice.
    #[must_use]
    pub fn from_points_pca(points: &[Vec3]) -> Self {
        assert!(!points.is_empty(), "Obb::from_points_pca requires points");
        let n = points.len() as f64;
        let mean = points.iter().fold(Vec3::ZERO, |a, &p| a + p) * (1.0 / n);
        let mut cov = [[0.0; 3]; 3];
        for p in points {
            let d = *p - mean;
            let v = [d.x, d.y, d.z];
            for (i, &vi) in v.iter().enumerate() {
                for (j, &vj) in v.iter().enumerate() {
                    cov[i][j] += vi * vj / n;
                }
            }
        }
        let cov_m = Mat3 { data: cov };
        let (_vals, axes) = cov_m.principal_axes_3x3();
        // Project all points to find extents along each axis.
        let ax = [
            Vec3::new(axes.data[0][0], axes.data[1][0], axes.data[2][0]),
            Vec3::new(axes.data[0][1], axes.data[1][1], axes.data[2][1]),
            Vec3::new(axes.data[0][2], axes.data[1][2], axes.data[2][2]),
        ];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            let d = *p - mean;
            for k in 0..3 {
                let proj = d.dot(&ax[k]);
                lo[k] = lo[k].min(proj);
                hi[k] = hi[k].max(proj);
            }
        }
        let center = mean
            + ax[0] * (0.5 * (lo[0] + hi[0]))
            + ax[1] * (0.5 * (lo[1] + hi[1]))
            + ax[2] * (0.5 * (lo[2] + hi[2]));
        Self {
            center,
            half_extents: Vec3::new(
                0.5 * (hi[0] - lo[0]),
                0.5 * (hi[1] - lo[1]),
                0.5 * (hi[2] - lo[2]),
            ),
            rotation: axes,
        }
    }
}

/// 3-D triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
}

impl Triangle {
    /// Unit normal of the CCW winding (ZERO for degenerate triangles).
    #[must_use]
    pub fn normal(&self) -> Vec3 {
        (self.b - self.a).cross(&(self.c - self.a)).normalized()
    }

    /// Area = ½|AB × AC|.
    #[must_use]
    pub fn area(&self) -> f64 {
        0.5 * (self.b - self.a).cross(&(self.c - self.a)).magnitude()
    }

    /// Centroid (A + B + C)/3.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        (self.a + self.b + self.c) * (1.0 / 3.0)
    }

    /// Barycentric coordinates (u, v, w) of the projection of p onto
    /// the triangle's plane, with u + v + w = 1 and p ≈ u·A + v·B + w·C
    /// (Ericson, *Real-Time Collision Detection*, §3.4).
    #[must_use]
    pub fn barycentric(&self, p: Vec3) -> (f64, f64, f64) {
        let v0 = self.b - self.a;
        let v1 = self.c - self.a;
        let v2 = p - self.a;
        let d00 = v0.dot(&v0);
        let d01 = v0.dot(&v1);
        let d11 = v1.dot(&v1);
        let d20 = v2.dot(&v0);
        let d21 = v2.dot(&v1);
        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1e-30 {
            return (1.0, 0.0, 0.0); // degenerate: everything maps to A
        }
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        (1.0 - v - w, v, w)
    }

    /// Point at barycentric coordinates (u, v, w).
    #[must_use]
    pub fn from_barycentric(&self, u: f64, v: f64, w: f64) -> Vec3 {
        self.a * u + self.b * v + self.c * w
    }

    /// Circumcenter (equidistant from the three vertices).
    #[must_use]
    pub fn circumcenter(&self) -> Vec3 {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        let n = ab.cross(&ac);
        let denom = 2.0 * n.magnitude_squared();
        if denom < 1e-30 {
            return self.centroid();
        }
        let t = (n.cross(&ab) * ac.magnitude_squared()
            + ac.cross(&n) * ab.magnitude_squared())
            * (1.0 / denom);
        self.a + t
    }

    /// Incenter, weighted by opposite side lengths.
    #[must_use]
    pub fn incenter(&self) -> Vec3 {
        let la = self.b.distance_to(&self.c);
        let lb = self.c.distance_to(&self.a);
        let lc = self.a.distance_to(&self.b);
        let s = la + lb + lc;
        if s < 1e-30 {
            return self.centroid();
        }
        (self.a * la + self.b * lb + self.c * lc) * (1.0 / s)
    }

    /// Degeneracy test: area below `tol`.
    #[must_use]
    pub fn is_degenerate(&self, tol: f64) -> bool {
        self.area() <= tol
    }

    /// Supporting plane (`None` for degenerate triangles).
    #[must_use]
    pub fn to_plane(&self) -> Option<Plane> {
        Plane::from_three_points(self.a, self.b, self.c)
    }
}

/// 2-D triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle2 {
    pub a: Vec2,
    pub b: Vec2,
    pub c: Vec2,
}

impl Triangle2 {
    /// Signed area (positive when CCW).
    #[must_use]
    pub fn area_signed(&self) -> f64 {
        0.5 * (self.b - self.a).cross(&(self.c - self.a))
    }

    /// Centroid.
    #[must_use]
    pub fn centroid(&self) -> Vec2 {
        (self.a + self.b + self.c) * (1.0 / 3.0)
    }

    /// Barycentric coordinates of p.
    #[must_use]
    pub fn barycentric(&self, p: Vec2) -> (f64, f64, f64) {
        let denom = (self.b - self.a).cross(&(self.c - self.a));
        if denom.abs() < 1e-30 {
            return (1.0, 0.0, 0.0);
        }
        let v = (p - self.a).cross(&(self.c - self.a)) / denom;
        let w = (self.b - self.a).cross(&(p - self.a)) / denom;
        (1.0 - v - w, v, w)
    }

    /// Circumcircle through the three vertices.
    ///
    /// # Panics
    /// Panics on collinear vertices.
    #[must_use]
    pub fn circumcircle(&self) -> Circle {
        let d = 2.0
            * (self.a.x * (self.b.y - self.c.y)
                + self.b.x * (self.c.y - self.a.y)
                + self.c.x * (self.a.y - self.b.y));
        assert!(d.abs() > 1e-14, "circumcircle requires non-collinear vertices");
        let a2 = self.a.magnitude_squared();
        let b2 = self.b.magnitude_squared();
        let c2 = self.c.magnitude_squared();
        let ux = (a2 * (self.b.y - self.c.y) + b2 * (self.c.y - self.a.y)
            + c2 * (self.a.y - self.b.y))
            / d;
        let uy = (a2 * (self.c.x - self.b.x) + b2 * (self.a.x - self.c.x)
            + c2 * (self.b.x - self.a.x))
            / d;
        let center = Vec2::new(ux, uy);
        Circle { center, radius: center.distance_to(&self.a) }
    }

    /// True when the winding is counter-clockwise.
    #[must_use]
    pub fn is_ccw(&self) -> bool {
        self.area_signed() > 0.0
    }
}

/// Capsule: segment swept by a sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f64,
}

/// Finite cylinder between two cap centers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f64,
}

/// Simple polygon in the plane (implicitly closed).
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon2 {
    pub vertices: Vec<Vec2>,
}

impl Polygon2 {
    /// # Panics
    /// Panics with fewer than 3 vertices.
    #[must_use]
    pub fn new(vertices: Vec<Vec2>) -> Self {
        assert!(vertices.len() >= 3, "Polygon2 requires at least 3 vertices");
        Self { vertices }
    }

    /// Shoelace signed area (positive when CCW).
    #[must_use]
    pub fn area_signed(&self) -> f64 {
        let n = self.vertices.len();
        let mut s = 0.0;
        for i in 0..n {
            let p = self.vertices[i];
            let q = self.vertices[(i + 1) % n];
            s += p.cross(&q);
        }
        0.5 * s
    }

    /// Absolute area.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.area_signed().abs()
    }

    /// Boundary length.
    #[must_use]
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        (0..n)
            .map(|i| self.vertices[i].distance_to(&self.vertices[(i + 1) % n]))
            .sum()
    }

    /// Area centroid (falls back to the vertex mean for degenerate
    /// zero-area polygons).
    #[must_use]
    pub fn centroid(&self) -> Vec2 {
        let n = self.vertices.len();
        let a = self.area_signed();
        if a.abs() < 1e-30 {
            let sum = self.vertices.iter().fold(Vec2::ZERO, |acc, &p| acc + p);
            return sum * (1.0 / n as f64);
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for i in 0..n {
            let p = self.vertices[i];
            let q = self.vertices[(i + 1) % n];
            let w = p.cross(&q);
            cx += (p.x + q.x) * w;
            cy += (p.y + q.y) * w;
        }
        Vec2::new(cx / (6.0 * a), cy / (6.0 * a))
    }

    /// Convexity: all cross products of consecutive edges share a sign
    /// (collinear runs allowed).
    #[must_use]
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        let mut sign = 0.0_f64;
        for i in 0..n {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % n];
            let c = self.vertices[(i + 2) % n];
            let cr = (b - a).cross(&(c - b));
            if cr.abs() > 1e-12 {
                if sign != 0.0 && cr.signum() != sign {
                    return false;
                }
                sign = cr.signum();
            }
        }
        true
    }

    /// True when the vertex order is counter-clockwise.
    #[must_use]
    pub fn is_ccw(&self) -> bool {
        self.area_signed() > 0.0
    }

    /// Reverses the winding in place.
    pub fn reverse(&mut self) {
        self.vertices.reverse();
    }

    /// Axis-aligned bounding rectangle.
    #[must_use]
    pub fn bounding_rect(&self) -> Rect {
        Rect::from_points(&self.vertices)
    }

    /// Simplicity: no two non-adjacent edges intersect (O(n²) test).
    #[must_use]
    pub fn is_simple(&self) -> bool {
        let n = self.vertices.len();
        let seg = |i: usize| (self.vertices[i], self.vertices[(i + 1) % n]);
        let intersects = |p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2| -> bool {
            let d1 = (p2 - p1).cross(&(p3 - p1));
            let d2 = (p2 - p1).cross(&(p4 - p1));
            let d3 = (p4 - p3).cross(&(p1 - p3));
            let d4 = (p4 - p3).cross(&(p2 - p3));
            (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
        };
        for i in 0..n {
            for j in (i + 1)..n {
                // Skip adjacent edges (sharing a vertex).
                if j == i || (j + 1) % n == i || (i + 1) % n == j {
                    continue;
                }
                let (a1, a2) = seg(i);
                let (b1, b2) = seg(j);
                if intersects(a1, a2, b1, b2) {
                    return false;
                }
            }
        }
        true
    }
}

/// 3-D polyline (open or closed).
#[derive(Debug, Clone, PartialEq)]
pub struct Polyline {
    pub points: Vec<Vec3>,
    pub closed: bool,
}

impl Polyline {
    /// Number of segments (accounting for closure).
    #[must_use]
    pub fn segment_count(&self) -> usize {
        if self.points.len() < 2 {
            0
        } else if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        }
    }

    /// Endpoints of segment i.
    #[must_use]
    pub fn segment(&self, i: usize) -> Segment {
        let n = self.points.len();
        Segment { a: self.points[i], b: self.points[(i + 1) % n] }
    }

    /// Total arclength.
    #[must_use]
    pub fn length(&self) -> f64 {
        (0..self.segment_count())
            .map(|i| {
                let s = self.segment(i);
                s.a.distance_to(&s.b)
            })
            .sum()
    }

    /// Point at arclength s (clamped to [0, length]).
    #[must_use]
    pub fn point_at_arclength(&self, s: f64) -> Vec3 {
        let mut remaining = s.max(0.0);
        for i in 0..self.segment_count() {
            let seg = self.segment(i);
            let len = seg.a.distance_to(&seg.b);
            if remaining <= len || i == self.segment_count() - 1 {
                let t = if len > 0.0 { (remaining / len).min(1.0) } else { 0.0 };
                return seg.a.lerp(&seg.b, t);
            }
            remaining -= len;
        }
        *self.points.last().unwrap_or(&Vec3::ZERO)
    }

    /// Unit tangent of the segment containing arclength s.
    #[must_use]
    pub fn tangent_at(&self, s: f64) -> Vec3 {
        let mut remaining = s.max(0.0);
        for i in 0..self.segment_count() {
            let seg = self.segment(i);
            let len = seg.a.distance_to(&seg.b);
            if remaining <= len || i == self.segment_count() - 1 {
                return (seg.b - seg.a).normalized();
            }
            remaining -= len;
        }
        Vec3::ZERO
    }

    /// Resamples at (approximately) uniform arclength spacing,
    /// preserving the endpoints.
    ///
    /// # Panics
    /// Panics unless spacing > 0.
    #[must_use]
    pub fn resample(&self, spacing: f64) -> Polyline {
        assert!(spacing > 0.0, "resample requires spacing > 0");
        let total = self.length();
        if total == 0.0 || self.points.len() < 2 {
            return self.clone();
        }
        let steps = (total / spacing).ceil().max(1.0) as usize;
        let mut points = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let s = total * i as f64 / steps as f64;
            points.push(self.point_at_arclength(s));
        }
        if self.closed {
            points.pop(); // last point duplicates the first
        }
        Polyline { points, closed: self.closed }
    }

    /// Axis-aligned bounds of the points.
    ///
    /// # Panics
    /// Panics on an empty polyline.
    #[must_use]
    pub fn bounding_box(&self) -> Aabb {
        Aabb::from_points(&self.points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::PI;

    #[test]
    fn test_plane_projection_and_distance() {
        let pl = Plane::from_point_normal(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        assert!((pl.signed_distance(Vec3::new(5.0, 3.0, 1.0)) - 2.0).abs() < 1e-12);
        let proj = pl.project(Vec3::new(5.0, 3.0, 1.0));
        assert!((proj.y - 1.0).abs() < 1e-12);
        assert!((pl.flip().signed_distance(Vec3::new(0.0, 3.0, 0.0)) + 2.0).abs() < 1e-12);
        assert!(Plane::from_three_points(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn test_aabb_ops() {
        let a = Aabb { min: Vec3::ZERO, max: Vec3::new(2.0, 2.0, 2.0) };
        let b = Aabb { min: Vec3::new(1.0, 1.0, 1.0), max: Vec3::new(3.0, 3.0, 3.0) };
        let u = a.union(&b);
        assert!(u.contains_point(Vec3::ZERO) && u.contains_point(Vec3::new(3.0, 3.0, 3.0)));
        let i = a.intersection(&b).unwrap();
        assert_eq!(i.min, Vec3::new(1.0, 1.0, 1.0));
        assert!(a
            .intersection(&Aabb { min: Vec3::new(5.0, 5.0, 5.0), max: Vec3::new(6.0, 6.0, 6.0) })
            .is_none());
        assert!((a.surface_area() - 24.0).abs() < 1e-12);
        assert!((a.volume() - 8.0).abs() < 1e-12);
        assert_eq!(a.corners().len(), 8);
        let t = a.transform(&Mat4::translation(Vec3::new(1.0, 0.0, 0.0)));
        assert_eq!(t.min, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_rect_ops() {
        let r = Rect { min: Vec2::ZERO, max: Vec2::new(4.0, 2.0) };
        assert!((r.area() - 8.0).abs() < 1e-12);
        assert!((r.perimeter() - 12.0).abs() < 1e-12);
        assert!(r.contains_point(Vec2::new(1.0, 1.0)));
        assert!(!r.contains_point(Vec2::new(5.0, 1.0)));
        assert_eq!(r.center(), Vec2::new(2.0, 1.0));
    }

    #[test]
    fn test_triangle_barycentric_roundtrip() {
        let t = Triangle {
            a: Vec3::new(0.0, 0.0, 0.0),
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 3.0, 0.0),
        };
        let p = t.from_barycentric(0.2, 0.3, 0.5);
        let (u, v, w) = t.barycentric(p);
        assert!((u - 0.2).abs() < 1e-12 && (v - 0.3).abs() < 1e-12 && (w - 0.5).abs() < 1e-12);
        assert!((t.area() - 3.0).abs() < 1e-12);
        assert!(t.normal().distance_to(&Vec3::new(0.0, 0.0, 1.0)) < 1e-12);
    }

    #[test]
    fn test_triangle_centers() {
        let t = Triangle {
            a: Vec3::new(0.0, 0.0, 0.0),
            b: Vec3::new(2.0, 0.0, 0.0),
            c: Vec3::new(0.0, 2.0, 0.0),
        };
        let cc = t.circumcenter();
        // Right triangle: circumcenter at hypotenuse midpoint.
        assert!(cc.distance_to(&Vec3::new(1.0, 1.0, 0.0)) < 1e-12);
        let ic = t.incenter();
        let r = 2.0 + 2.0 - 8.0_f64.sqrt(); // 2·inradius for this triangle
        assert!((ic.x - r / 2.0).abs() < 1e-12 && (ic.y - r / 2.0).abs() < 1e-12);
        assert!(!t.is_degenerate(1e-12));
        assert!(t.to_plane().is_some());
    }

    #[test]
    fn test_triangle2_circumcircle_and_orientation() {
        let t = Triangle2 {
            a: Vec2::new(0.0, 0.0),
            b: Vec2::new(2.0, 0.0),
            c: Vec2::new(1.0, 2.0),
        };
        assert!(t.is_ccw());
        let cc = t.circumcircle();
        for p in [t.a, t.b, t.c] {
            assert!((cc.center.distance_to(&p) - cc.radius).abs() < 1e-12);
        }
        let (u, v, w) = t.barycentric(t.centroid());
        assert!((u - 1.0 / 3.0).abs() < 1e-12 && (v - 1.0 / 3.0).abs() < 1e-12 && (w - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_polygon_regular_ngon_area() {
        // Regular hexagon with side length 1 matches the closed form.
        let n = 6;
        let side: f64 = 1.0;
        let circumradius = side / (2.0 * (PI / n as f64).sin());
        let verts: Vec<Vec2> = (0..n)
            .map(|i| {
                let a = 2.0 * PI * i as f64 / n as f64;
                Vec2::new(circumradius * a.cos(), circumradius * a.sin())
            })
            .collect();
        let poly = Polygon2::new(verts);
        let expected = crate::geometry::area_regular_polygon(n as u32, side);
        assert!((poly.area() - expected).abs() < 1e-10);
        assert!(poly.is_convex());
        assert!(poly.is_ccw());
        assert!(poly.is_simple());
        assert!((poly.perimeter() - 6.0).abs() < 1e-10);
        assert!(poly.centroid().magnitude() < 1e-12);
    }

    #[test]
    fn test_polygon_nonconvex_and_nonsimple() {
        let star = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(2.0, 3.0),
        ]);
        assert!(!star.is_convex());
        assert!(star.is_simple());
        // Self-intersecting bowtie.
        let bowtie = Polygon2::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        ]);
        assert!(!bowtie.is_simple());
    }

    #[test]
    fn test_obb_roundtrip_and_pca() {
        let rot = crate::linalg::rotation_z(0.5);
        let obb = Obb {
            center: Vec3::new(1.0, 2.0, 3.0),
            half_extents: Vec3::new(2.0, 1.0, 0.5),
            rotation: rot,
        };
        let aabb = obb.to_aabb();
        for c in obb.corners() {
            assert!(aabb.expand(1e-12).contains_point(c));
        }
        // PCA box on an axis-aligned cloud recovers roughly the cloud extents.
        let mut pts = Vec::new();
        for i in 0..20 {
            for j in 0..5 {
                pts.push(Vec3::new(i as f64 * 0.5, j as f64 * 0.2, 0.0));
            }
        }
        let fitted = Obb::from_points_pca(&pts);
        let vol_box = 8.0 * fitted.half_extents.x.max(1e-9)
            * fitted.half_extents.y.max(1e-9)
            * fitted.half_extents.z.max(1e-9);
        assert!(fitted.half_extents.x >= fitted.half_extents.y);
        assert!(vol_box < 10.0, "PCA box should be tight, got volume {vol_box}");
    }

    #[test]
    fn test_polyline_arclength_and_resample() {
        let pl = Polyline {
            points: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            closed: false,
        };
        assert!((pl.length() - 2.0).abs() < 1e-12);
        let mid = pl.point_at_arclength(1.5);
        assert!(mid.distance_to(&Vec3::new(1.0, 0.5, 0.0)) < 1e-12);
        assert!(pl.tangent_at(0.5).distance_to(&Vec3::new(1.0, 0.0, 0.0)) < 1e-12);
        assert!(pl.tangent_at(1.5).distance_to(&Vec3::new(0.0, 1.0, 0.0)) < 1e-12);
        let rs = pl.resample(0.5);
        assert_eq!(rs.points.len(), 5);
        assert!((rs.length() - 2.0).abs() < 1e-9);
        // Closed polyline includes the wrap segment.
        let sq = Polyline {
            points: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            closed: true,
        };
        assert!((sq.length() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_rect_intersection_and_union() {
        let a = Rect { min: Vec2::ZERO, max: Vec2::new(4.0, 2.0) };
        let b = Rect { min: Vec2::new(1.0, 1.0), max: Vec2::new(6.0, 5.0) };
        // Overlapping: intersection is the componentwise max/min box.
        let i = a.intersection(&b).unwrap();
        assert_eq!(i.min, Vec2::new(1.0, 1.0));
        assert_eq!(i.max, Vec2::new(4.0, 2.0));
        assert!((i.area() - 3.0).abs() < 1e-12);
        // The intersection is symmetric and contained in both inputs.
        assert_eq!(b.intersection(&a).unwrap(), i);
        for c in i.corners() {
            assert!(a.contains_point(c) && b.contains_point(c));
        }
        // Disjoint rectangles have no intersection.
        let far = Rect { min: Vec2::new(10.0, 10.0), max: Vec2::new(11.0, 11.0) };
        assert!(a.intersection(&far).is_none());
        // Union: smallest rectangle containing both.
        let u = a.union(&b);
        assert_eq!(u.min, Vec2::ZERO);
        assert_eq!(u.max, Vec2::new(6.0, 5.0));
        for r in [&a, &b] {
            for c in r.corners() {
                assert!(u.contains_point(c));
            }
        }
        // Inclusion-exclusion lower bound: |A ∪ B| box ≥ |A| + |B| − |A ∩ B|.
        assert!(u.area() >= a.area() + b.area() - i.area() - 1e-12);
        // Union with a disjoint box still bounds both.
        let ud = a.union(&far);
        assert_eq!(ud.min, Vec2::ZERO);
        assert_eq!(ud.max, Vec2::new(11.0, 11.0));
    }

    #[test]
    fn test_rect_expand_and_extents() {
        let a = Rect { min: Vec2::new(-1.0, 0.0), max: Vec2::new(3.0, 2.0) };
        let e = a.expand(0.5);
        // Grown by the margin on every side.
        assert_eq!(e.min, Vec2::new(-1.5, -0.5));
        assert_eq!(e.max, Vec2::new(3.5, 2.5));
        // Expansion contains the original corners; area grows by
        // (w + 2m)(h + 2m) − wh.
        for c in a.corners() {
            assert!(e.contains_point(c));
        }
        assert!((e.area() - 5.0_f64 * 3.0_f64).abs() < 1e-12);
        // Extents are half-widths: center ± extents recovers min/max.
        let ext = a.extents();
        assert_eq!(ext, Vec2::new(2.0, 1.0));
        assert_eq!(a.center() - ext, a.min);
        assert_eq!(a.center() + ext, a.max);
        // Area equals the product of the full widths 2·ex · 2·ey.
        assert!((a.area() - 4.0 * ext.x * ext.y).abs() < 1e-12);
    }

    #[test]
    fn test_polyline_bounding_box() {
        let pl = Polyline {
            points: vec![
                Vec3::new(1.0, -2.0, 0.5),
                Vec3::new(-3.0, 4.0, 2.0),
                Vec3::new(2.0, 1.0, -1.0),
                Vec3::new(0.0, 0.0, 0.0),
            ],
            closed: false,
        };
        let bb = pl.bounding_box();
        // Contains every vertex.
        for &p in &pl.points {
            assert!(bb.contains_point(p));
        }
        // Tight: every face of the box touches some vertex.
        assert_eq!(bb.min, Vec3::new(-3.0, -2.0, -1.0));
        assert_eq!(bb.max, Vec3::new(2.0, 4.0, 2.0));
        for axis in 0..3 {
            let get = |v: Vec3| [v.x, v.y, v.z][axis];
            assert!(pl.points.iter().any(|&p| (get(p) - get(bb.min)).abs() < 1e-15));
            assert!(pl.points.iter().any(|&p| (get(p) - get(bb.max)).abs() < 1e-15));
        }
    }

    #[test]
    fn test_ray_at() {
        let r = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0));
        assert!(r.at(2.0).distance_to(&Vec3::new(1.0, 2.0, 0.0)) < 1e-12);
        assert!((r.dir.magnitude() - 1.0).abs() < 1e-12);
    }
}
