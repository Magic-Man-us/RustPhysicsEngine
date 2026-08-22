//! Homogeneous 2-D projective geometry: points, lines, cross ratios,
//! and plane homographies (Hartley & Zisserman, *Multiple View
//! Geometry*, ch. 2 and 4).

use crate::math::Vec2;

const EPS: f64 = 1e-12;

/// Homogeneous 2-D coordinates: a point (x, y, w) or a line
/// (a, b, c) with ax + by + c = 0.
pub type Hom2 = [f64; 3];

/// Lifts a Euclidean point to homogeneous coordinates (w = 1).
#[must_use]
pub fn point_h(p: Vec2) -> Hom2 {
    [p.x, p.y, 1.0]
}

/// Projects back to Euclidean coordinates; `None` for points at
/// infinity (w ≈ 0).
#[must_use]
pub fn dehomogenize(h: Hom2) -> Option<Vec2> {
    if h[2].abs() < EPS {
        None
    } else {
        Some(Vec2::new(h[0] / h[2], h[1] / h[2]))
    }
}

fn cross3(a: Hom2, b: Hom2) -> Hom2 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The line through two points: l = a × b.
#[must_use]
pub fn line_through(a: Vec2, b: Vec2) -> Hom2 {
    cross3(point_h(a), point_h(b))
}

/// Intersection of two lines: p = l₁ × l₂ (possibly at infinity for
/// parallel lines).
#[must_use]
pub fn lines_intersect(l1: Hom2, l2: Hom2) -> Hom2 {
    cross3(l1, l2)
}

/// Incidence test: p·l = 0 within tol (both scale-normalized).
#[must_use]
pub fn point_on_line(p: Hom2, l: Hom2, tol: f64) -> bool {
    let pn = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let ln = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
    if pn < EPS || ln < EPS {
        return false;
    }
    ((p[0] * l[0] + p[1] * l[1] + p[2] * l[2]) / (pn * ln)).abs() <= tol
}

/// Collinearity of three Euclidean points (twice the triangle area
/// below tol, scale-normalized).
#[must_use]
pub fn are_collinear(a: Vec2, b: Vec2, c: Vec2, tol: f64) -> bool {
    let area2 = (b - a).cross(&(c - a));
    let scale = (b - a).magnitude().max((c - a).magnitude()).max(1.0);
    area2.abs() <= tol * scale * scale
}

/// Cross ratio of four collinear parameters:
/// (a, b; c, d) = ((a−c)(b−d)) / ((a−d)(b−c)).
///
/// # Panics
/// Panics when the denominator vanishes (repeated points).
#[must_use]
pub fn cross_ratio(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let denom = (a - d) * (b - c);
    assert!(denom.abs() > EPS, "cross_ratio requires distinct points");
    ((a - c) * (b - d)) / denom
}

/// Plane projective transform p' ~ H·p.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    pub h: [[f64; 3]; 3],
}

/// Local Gaussian elimination with partial pivoting for the 8×8 DLT
/// system (kept self-contained: projective needs no Part-1 machinery).
fn solve_gauss8(mut a: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut piv = col;
        for r in (col + 1)..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        for r in (col + 1)..8 {
            let f = a[r][col] / a[col][col];
            for c in col..9 {
                a[r][c] -= f * a[col][c];
            }
        }
    }
    let mut x = [0.0; 8];
    for i in (0..8).rev() {
        let mut s = a[i][8];
        for j in (i + 1)..8 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

impl Homography {
    /// Direct linear transform from four point correspondences
    /// (h₃₃ normalized to 1). `None` for degenerate configurations
    /// (three collinear source or destination points).
    #[must_use]
    pub fn from_four_points(src: [Vec2; 4], dst: [Vec2; 4]) -> Option<Self> {
        let mut a = [[0.0; 9]; 8];
        for i in 0..4 {
            let (x, y) = (src[i].x, src[i].y);
            let (xp, yp) = (dst[i].x, dst[i].y);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, xp];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y, yp];
        }
        let h = solve_gauss8(a)?;
        Some(Self {
            h: [
                [h[0], h[1], h[2]],
                [h[3], h[4], h[5]],
                [h[6], h[7], 1.0],
            ],
        })
    }

    /// Applies to a Euclidean point; `None` when the image lies at
    /// infinity.
    #[must_use]
    pub fn apply(&self, p: Vec2) -> Option<Vec2> {
        let hp = [
            self.h[0][0] * p.x + self.h[0][1] * p.y + self.h[0][2],
            self.h[1][0] * p.x + self.h[1][1] * p.y + self.h[1][2],
            self.h[2][0] * p.x + self.h[2][1] * p.y + self.h[2][2],
        ];
        dehomogenize(hp)
    }

    /// Inverse homography via the adjugate; `None` when singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.h;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < EPS {
            return None;
        }
        let inv = 1.0 / det;
        let adj = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
            ],
        ];
        Some(Self { h: adj })
    }

    /// Composition self ∘ other (apply `other` first).
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| self.h[i][k] * other.h[k][j]).sum();
            }
        }
        Self { h: out }
    }

    /// Image of the point at infinity in the given direction — the
    /// vanishing point of all lines with that direction. `None` when
    /// the direction stays at infinity (affine maps).
    #[must_use]
    pub fn vanishing_point(&self, direction: Vec2) -> Option<Vec2> {
        let hp = [
            self.h[0][0] * direction.x + self.h[0][1] * direction.y,
            self.h[1][0] * direction.x + self.h[1][1] * direction.y,
            self.h[2][0] * direction.x + self.h[2][1] * direction.y,
        ];
        dehomogenize(hp)
    }
}

/// Homography mapping an arbitrary quad (CCW or CW consistent order)
/// onto the axis-aligned rectangle [0, w]×[0, h] with corner order
/// (0,0), (w,0), (w,h), (0,h).
#[must_use]
pub fn rectify_quad_to_rect(quad: [Vec2; 4], width: f64, height: f64) -> Option<Homography> {
    Homography::from_four_points(
        quad,
        [
            Vec2::new(0.0, 0.0),
            Vec2::new(width, 0.0),
            Vec2::new(width, height),
            Vec2::new(0.0, height),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_line_incidence() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 2.0);
        let l = line_through(a, b);
        assert!(point_on_line(point_h(Vec2::new(1.0, 1.0)), l, 1e-12));
        assert!(!point_on_line(point_h(Vec2::new(1.0, 0.0)), l, 1e-9));
    }

    #[test]
    fn test_lines_intersect_and_parallel() {
        let l1 = line_through(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let l2 = line_through(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        let p = dehomogenize(lines_intersect(l1, l2)).unwrap();
        assert!(p.distance_to(&Vec2::new(0.5, 0.5)) < 1e-12);
        // Parallel lines meet at infinity.
        let l3 = line_through(Vec2::new(0.0, 1.0), Vec2::new(1.0, 2.0));
        assert!(dehomogenize(lines_intersect(l1, l3)).is_none());
    }

    #[test]
    fn test_collinear_and_cross_ratio() {
        assert!(are_collinear(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(2.0, 4.0),
            1e-12
        ));
        assert!(!are_collinear(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(2.0, 4.1),
            1e-9
        ));
        // Harmonic set: (0, 2; 1, 4) ... classical harmonic quadruple
        // (a, b; c, d) = -1 for a=0, b=2, c=1, d at infinity is a limit;
        // use a concrete identity instead:
        let cr = cross_ratio(0.0, 3.0, 1.0, 2.0);
        assert!((cr - ((0.0 - 1.0) * (3.0 - 2.0)) / ((0.0 - 2.0) * (3.0 - 1.0))).abs() < 1e-15);
    }

    #[test]
    fn test_homography_maps_four_points() {
        let src = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let dst = [
            Vec2::new(0.1, -0.2),
            Vec2::new(1.4, 0.3),
            Vec2::new(1.1, 1.2),
            Vec2::new(-0.3, 0.9),
        ];
        let h = Homography::from_four_points(src, dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            assert!(h.apply(*s).unwrap().distance_to(d) < 1e-10);
        }
        let inv = h.inverse().unwrap();
        for (s, d) in src.iter().zip(&dst) {
            assert!(inv.apply(*d).unwrap().distance_to(s) < 1e-9);
        }
    }

    #[test]
    fn test_homography_preserves_collinearity() {
        let src = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let dst = [
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.5),
            Vec2::new(2.5, 2.8),
            Vec2::new(-0.5, 2.0),
        ];
        let h = Homography::from_four_points(src, dst).unwrap();
        let a = h.apply(Vec2::new(0.5, 0.5)).unwrap();
        let b = h.apply(Vec2::new(1.0, 1.0)).unwrap();
        let c = h.apply(Vec2::new(1.5, 1.5)).unwrap();
        assert!(are_collinear(a, b, c, 1e-9));
    }

    #[test]
    fn test_cross_ratio_invariant_under_homography() {
        let src = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let dst = [
            Vec2::new(0.2, 0.1),
            Vec2::new(2.6, -0.4),
            Vec2::new(3.0, 2.5),
            Vec2::new(-0.4, 1.8),
        ];
        let h = Homography::from_four_points(src, dst).unwrap();
        // Four collinear points on y = 0.7 x with parameters t.
        let ts = [0.2, 0.6, 1.1, 1.9];
        let dir = Vec2::new(1.0, 0.7);
        let images: Vec<Vec2> = ts.iter().map(|&t| h.apply(dir * t).unwrap()).collect();
        // Parameterize images along their common line by projection.
        let axis = (images[3] - images[0]).normalized();
        let params: Vec<f64> = images.iter().map(|p| (*p - images[0]).dot(&axis)).collect();
        let cr_src = cross_ratio(ts[0], ts[1], ts[2], ts[3]);
        let cr_dst = cross_ratio(params[0], params[1], params[2], params[3]);
        assert!((cr_src - cr_dst).abs() < 1e-9, "{cr_src} vs {cr_dst}");
    }

    #[test]
    fn test_rectify_quad() {
        let quad = [
            Vec2::new(0.2, 0.1),
            Vec2::new(2.3, 0.4),
            Vec2::new(2.0, 1.8),
            Vec2::new(-0.1, 1.5),
        ];
        let h = rectify_quad_to_rect(quad, 4.0, 3.0).unwrap();
        assert!(h.apply(quad[0]).unwrap().distance_to(&Vec2::new(0.0, 0.0)) < 1e-9);
        assert!(h.apply(quad[2]).unwrap().distance_to(&Vec2::new(4.0, 3.0)) < 1e-9);
    }

    #[test]
    fn test_vanishing_point() {
        // A perspective map with nonzero bottom row sends some
        // directions to finite vanishing points.
        let src = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let dst = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.8, 0.8),
            Vec2::new(0.2, 0.8),
        ];
        let h = Homography::from_four_points(src, dst).unwrap();
        let vp = h.vanishing_point(Vec2::new(0.0, 1.0));
        assert!(vp.is_some(), "perspective map should give a finite vanishing point");
        // An affine (identity) map keeps directions at infinity.
        let id = Homography { h: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
        assert!(id.vanishing_point(Vec2::new(1.0, 0.0)).is_none());
    }
}
