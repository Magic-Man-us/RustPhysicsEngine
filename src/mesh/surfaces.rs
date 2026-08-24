//! Parametric surfaces: Bézier/B-spline/NURBS patches, classic surface
//! constructions, differential geometry via fundamental forms, and a
//! catalogue of named surfaces.

use crate::math::{Vec2, Vec3};
use crate::mesh::generate::from_parametric;
use crate::mesh::Mesh;

/// Bicubic Bézier patch; `control[i][j]` weights the Bernstein product
/// B_i(u) B_j(v).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierPatch {
    pub control: [[Vec3; 4]; 4],
}

fn bernstein3(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t]
}

/// Derivatives of the cubic Bernstein basis.
fn bernstein3_d(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [
        -3.0 * s * s,
        3.0 * s * s - 6.0 * s * t,
        6.0 * s * t - 3.0 * t * t,
        3.0 * t * t,
    ]
}

/// De Casteljau split of one cubic row at t = 1/2.
fn split_cubic(p: [Vec3; 4]) -> ([Vec3; 4], [Vec3; 4]) {
    let m01 = (p[0] + p[1]) * 0.5;
    let m12 = (p[1] + p[2]) * 0.5;
    let m23 = (p[2] + p[3]) * 0.5;
    let a = (m01 + m12) * 0.5;
    let b = (m12 + m23) * 0.5;
    let c = (a + b) * 0.5;
    ([p[0], m01, a, c], [c, b, m23, p[3]])
}

impl BezierPatch {
    /// Point at `(u, v)`, both in [0, 1].
    #[must_use]
    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        let (bu, bv) = (bernstein3(u), bernstein3(v));
        let mut p = Vec3::ZERO;
        for (i, row) in self.control.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                p = p + c * (bu[i] * bv[j]);
            }
        }
        p
    }

    /// Partial derivative in u.
    #[must_use]
    pub fn du(&self, u: f64, v: f64) -> Vec3 {
        let (bu, bv) = (bernstein3_d(u), bernstein3(v));
        let mut p = Vec3::ZERO;
        for (i, row) in self.control.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                p = p + c * (bu[i] * bv[j]);
            }
        }
        p
    }

    /// Partial derivative in v.
    #[must_use]
    pub fn dv(&self, u: f64, v: f64) -> Vec3 {
        let (bu, bv) = (bernstein3(u), bernstein3_d(v));
        let mut p = Vec3::ZERO;
        for (i, row) in self.control.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                p = p + c * (bu[i] * bv[j]);
            }
        }
        p
    }

    /// Unit normal `du × dv` (zero where the patch is degenerate).
    #[must_use]
    pub fn normal(&self, u: f64, v: f64) -> Vec3 {
        let n = self.du(u, v).cross(&self.dv(u, v));
        let m = n.magnitude();
        if m > 0.0 {
            n * (1.0 / m)
        } else {
            Vec3::ZERO
        }
    }

    /// Tessellates on an `nu` x `nv` cell grid.
    #[must_use]
    pub fn to_mesh(&self, nu: usize, nv: usize) -> Mesh {
        from_parametric(&|u, v| self.eval(u, v), (0.0, 1.0), (0.0, 1.0), nu, nv, false, false)
    }

    /// Splits into four subpatches at the parametric midpoint, ordered
    /// `[u-low v-low, u-high v-low, u-low v-high, u-high v-high]`.
    /// Their union reproduces the original surface exactly.
    #[must_use]
    pub fn subdivide(&self) -> [BezierPatch; 4] {
        // Split along u (rows indexed by i).
        let mut lo = [[Vec3::ZERO; 4]; 4];
        let mut hi = [[Vec3::ZERO; 4]; 4];
        for j in 0..4 {
            let col = [self.control[0][j], self.control[1][j], self.control[2][j], self.control[3][j]];
            let (a, b) = split_cubic(col);
            for i in 0..4 {
                lo[i][j] = a[i];
                hi[i][j] = b[i];
            }
        }
        // Split each half along v.
        let split_v = |half: &[[Vec3; 4]; 4]| {
            let mut first = [[Vec3::ZERO; 4]; 4];
            let mut second = [[Vec3::ZERO; 4]; 4];
            for i in 0..4 {
                let (a, b) = split_cubic(half[i]);
                first[i] = a;
                second[i] = b;
            }
            (BezierPatch { control: first }, BezierPatch { control: second })
        };
        let (ll, lh) = split_v(&lo);
        let (hl, hh) = split_v(&hi);
        [ll, hl, lh, hh]
    }
}

/// Cox-de Boor recursion for one B-spline basis value N_{i,p}(t).
fn bspline_basis(knots: &[f64], i: usize, p: usize, t: f64) -> f64 {
    if p == 0 {
        // Half-open spans; the final span is closed so the curve
        // reaches its last control point.
        let last = knots[i + 1] >= knots[knots.len() - 1];
        return if (knots[i] <= t && t < knots[i + 1]) || (last && t == knots[i + 1]) {
            1.0
        } else {
            0.0
        };
    }
    let mut acc = 0.0;
    let d1 = knots[i + p] - knots[i];
    if d1 > 0.0 {
        acc += (t - knots[i]) / d1 * bspline_basis(knots, i, p - 1, t);
    }
    let d2 = knots[i + p + 1] - knots[i + 1];
    if d2 > 0.0 {
        acc += (knots[i + p + 1] - t) / d2 * bspline_basis(knots, i + 1, p - 1, t);
    }
    acc
}

/// Clamped uniform knot vector for `n` control points, degree `p`:
/// domain [0, 1].
fn clamped_knots(n: usize, p: usize) -> Vec<f64> {
    let interior = n - p; // number of spans
    let mut knots = vec![0.0; p + 1];
    for k in 1..interior {
        knots.push(k as f64 / interior as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, p + 1));
    knots
}

/// Tensor-product B-spline surface. Parameters range over
/// `[knots[degree], knots[len - degree - 1]]` in each direction.
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineSurface {
    pub degree_u: usize,
    pub degree_v: usize,
    pub knots_u: Vec<f64>,
    pub knots_v: Vec<f64>,
    /// `control[i][j]`: i along u, j along v.
    pub control: Vec<Vec<Vec3>>,
}

impl BSplineSurface {
    /// Clamped surface with uniform interior knots on [0, 1]².
    ///
    /// # Panics
    /// Panics unless the control net is rectangular with more than
    /// `degree` points per direction and `degree >= 1`.
    #[must_use]
    pub fn uniform(degree_u: usize, degree_v: usize, control: Vec<Vec<Vec3>>) -> Self {
        assert!(degree_u >= 1 && degree_v >= 1, "degree must be >= 1");
        let nu = control.len();
        assert!(nu > degree_u, "need more than degree_u control rows");
        let nv = control[0].len();
        assert!(nv > degree_v, "need more than degree_v control columns");
        assert!(control.iter().all(|r| r.len() == nv), "ragged control net");
        Self {
            degree_u,
            degree_v,
            knots_u: clamped_knots(nu, degree_u),
            knots_v: clamped_knots(nv, degree_v),
            control,
        }
    }

    fn domain(&self) -> ((f64, f64), (f64, f64)) {
        (
            (
                self.knots_u[self.degree_u],
                self.knots_u[self.knots_u.len() - self.degree_u - 1],
            ),
            (
                self.knots_v[self.degree_v],
                self.knots_v[self.knots_v.len() - self.degree_v - 1],
            ),
        )
    }

    /// Point at `(u, v)` (clamped to the domain).
    #[must_use]
    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        let ((u0, u1), (v0, v1)) = self.domain();
        let u = u.clamp(u0, u1);
        let v = v.clamp(v0, v1);
        let mut p = Vec3::ZERO;
        for (i, row) in self.control.iter().enumerate() {
            let bu = bspline_basis(&self.knots_u, i, self.degree_u, u);
            if bu == 0.0 {
                continue;
            }
            for (j, &c) in row.iter().enumerate() {
                let bv = bspline_basis(&self.knots_v, j, self.degree_v, v);
                p = p + c * (bu * bv);
            }
        }
        p
    }

    /// Unit normal by central differences (steps shrink at the domain
    /// boundary).
    #[must_use]
    pub fn normal(&self, u: f64, v: f64) -> Vec3 {
        let ((u0, u1), (v0, v1)) = self.domain();
        let h = 1e-5 * (u1 - u0).max(v1 - v0);
        let du = self.eval((u + h).min(u1), v) - self.eval((u - h).max(u0), v);
        let dv = self.eval(u, (v + h).min(v1)) - self.eval(u, (v - h).max(v0));
        let n = du.cross(&dv);
        let m = n.magnitude();
        if m > 0.0 {
            n * (1.0 / m)
        } else {
            Vec3::ZERO
        }
    }

    /// Tessellates on an `nu` x `nv` cell grid over the whole domain.
    #[must_use]
    pub fn to_mesh(&self, nu: usize, nv: usize) -> Mesh {
        let ((u0, u1), (v0, v1)) = self.domain();
        from_parametric(&|u, v| self.eval(u, v), (u0, u1), (v0, v1), nu, nv, false, false)
    }
}

/// Tensor-product NURBS surface (rational B-spline): projective
/// weights allow exact conics.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsSurface {
    pub degree_u: usize,
    pub degree_v: usize,
    pub knots_u: Vec<f64>,
    pub knots_v: Vec<f64>,
    pub control: Vec<Vec<Vec3>>,
    pub weights: Vec<Vec<f64>>,
}

/// Control points and weights of the exact unit circle in the plane as
/// a quadratic NURBS of 9 control points (four 90° arcs), knots
/// [0,0,0,¼,¼,½,½,¾,¾,1,1,1].
fn circle_controls() -> ([Vec2; 9], [f64; 9], Vec<f64>) {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let pts = [
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(-1.0, 1.0),
        Vec2::new(-1.0, 0.0),
        Vec2::new(-1.0, -1.0),
        Vec2::new(0.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 0.0),
    ];
    let w = [1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0];
    let knots = vec![0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0];
    (pts, w, knots)
}

impl NurbsSurface {
    /// Point at `(u, v)`: rational combination
    /// Σ wᵢⱼ Nᵢ(u) Nⱼ(v) Pᵢⱼ / Σ wᵢⱼ Nᵢ(u) Nⱼ(v).
    #[must_use]
    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        let u = u.clamp(
            self.knots_u[self.degree_u],
            self.knots_u[self.knots_u.len() - self.degree_u - 1],
        );
        let v = v.clamp(
            self.knots_v[self.degree_v],
            self.knots_v[self.knots_v.len() - self.degree_v - 1],
        );
        let mut num = Vec3::ZERO;
        let mut den = 0.0;
        for (i, row) in self.control.iter().enumerate() {
            let bu = bspline_basis(&self.knots_u, i, self.degree_u, u);
            if bu == 0.0 {
                continue;
            }
            for (j, &c) in row.iter().enumerate() {
                let bv = bspline_basis(&self.knots_v, j, self.degree_v, v);
                let w = bu * bv * self.weights[i][j];
                num = num + c * w;
                den += w;
            }
        }
        num * (1.0 / den)
    }

    /// Tessellates on an `nu` x `nv` cell grid over the whole domain.
    #[must_use]
    pub fn to_mesh(&self, nu: usize, nv: usize) -> Mesh {
        let u0 = self.knots_u[self.degree_u];
        let u1 = self.knots_u[self.knots_u.len() - self.degree_u - 1];
        let v0 = self.knots_v[self.degree_v];
        let v1 = self.knots_v[self.knots_v.len() - self.degree_v - 1];
        from_parametric(&|u, v| self.eval(u, v), (u0, u1), (v0, v1), nu, nv, false, false)
    }

    /// Exact sphere of radius `r`: a semicircular profile revolved by
    /// the exact NURBS circle. Every evaluated point lies exactly on
    /// the sphere.
    ///
    /// # Panics
    /// Panics unless `r > 0`.
    #[must_use]
    pub fn sphere(r: f64) -> Self {
        assert!(r > 0.0, "sphere radius must be positive");
        let (circle, wc, knots_u) = circle_controls();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // Semicircle from south to north pole in the (radial, y) plane.
        let profile = [
            Vec2::new(0.0, -r),
            Vec2::new(r, -r),
            Vec2::new(r, 0.0),
            Vec2::new(r, r),
            Vec2::new(0.0, r),
        ];
        let wp = [1.0, s, 1.0, s, 1.0];
        let knots_v = vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0];
        let control = circle
            .iter()
            .map(|c| profile.iter().map(|p| Vec3::new(p.x * c.x, p.y, p.x * c.y)).collect())
            .collect();
        let weights =
            wc.iter().map(|a| wp.iter().map(|b| a * b).collect()).collect();
        Self { degree_u: 2, degree_v: 2, knots_u, knots_v, control, weights }
    }

    /// Exact torus: minor circle of radius `r` revolved at major
    /// radius `big_r` around the y axis.
    ///
    /// # Panics
    /// Panics unless `0 < r < big_r`.
    #[must_use]
    pub fn torus(big_r: f64, r: f64) -> Self {
        assert!(r > 0.0 && big_r > r, "torus requires 0 < r < R");
        let (circle, wc, knots) = circle_controls();
        let control: Vec<Vec<Vec3>> = circle
            .iter()
            .map(|c| {
                circle
                    .iter()
                    .map(|m| {
                        let radial = big_r + r * m.x;
                        Vec3::new(radial * c.x, r * m.y, radial * c.y)
                    })
                    .collect()
            })
            .collect();
        let weights = wc.iter().map(|a| wc.iter().map(|b| a * b).collect()).collect();
        Self {
            degree_u: 2,
            degree_v: 2,
            knots_u: knots.clone(),
            knots_v: knots,
            control,
            weights,
        }
    }

    /// Exact cylinder of radius `r` and height `h` along the y axis,
    /// base at y = 0.
    ///
    /// # Panics
    /// Panics unless `r > 0` and `h > 0`.
    #[must_use]
    pub fn cylinder(r: f64, h: f64) -> Self {
        assert!(r > 0.0 && h > 0.0, "cylinder requires positive radius and height");
        let (circle, wc, knots_u) = circle_controls();
        let control: Vec<Vec<Vec3>> = circle
            .iter()
            .map(|c| {
                vec![
                    Vec3::new(r * c.x, 0.0, r * c.y),
                    Vec3::new(r * c.x, h, r * c.y),
                ]
            })
            .collect();
        let weights = wc.iter().map(|a| vec![*a, *a]).collect();
        Self {
            degree_u: 2,
            degree_v: 1,
            knots_u,
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            control,
            weights,
        }
    }
}

/// Point of the surface of revolution of a planar profile
/// `t ↦ (radius, height)` rotated by `theta` around the y axis.
#[must_use]
pub fn surface_of_revolution(profile: &dyn Fn(f64) -> Vec2, t: f64, theta: f64) -> Vec3 {
    let p = profile(t);
    Vec3::new(p.x * theta.cos(), p.y, p.x * theta.sin())
}

/// Ruled surface: linear blend between two curves,
/// `(1 − v) c1(u) + v c2(u)`.
#[must_use]
pub fn ruled_surface(c1: &dyn Fn(f64) -> Vec3, c2: &dyn Fn(f64) -> Vec3, u: f64, v: f64) -> Vec3 {
    c1(u) * (1.0 - v) + c2(u) * v
}

/// Bilinearly blended Coons patch interpolating four boundary curves:
/// `c0` (v = 0), `c1` (v = 1) over u, and `d0` (u = 0), `d1` (u = 1)
/// over v. The curves must agree at the corners.
#[must_use]
pub fn coons_patch(
    c0: &dyn Fn(f64) -> Vec3,
    c1: &dyn Fn(f64) -> Vec3,
    d0: &dyn Fn(f64) -> Vec3,
    d1: &dyn Fn(f64) -> Vec3,
    u: f64,
    v: f64,
) -> Vec3 {
    let lc = c0(u) * (1.0 - v) + c1(u) * v;
    let ld = d0(v) * (1.0 - u) + d1(v) * u;
    let b = c0(0.0) * ((1.0 - u) * (1.0 - v))
        + c0(1.0) * (u * (1.0 - v))
        + c1(0.0) * ((1.0 - u) * v)
        + c1(1.0) * (u * v);
    lc + ld - b
}

/// First (E, F, G) and second (L, M, N) fundamental form coefficients
/// of a parametric surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundamentalForms {
    pub e: f64,
    pub f: f64,
    pub g: f64,
    pub l: f64,
    pub m: f64,
    pub n: f64,
}

/// Fundamental forms by central finite differences with step `h`.
///
/// # Panics
/// Panics unless `h > 0` and the surface is regular (nonzero
/// `f_u × f_v`) at `(u, v)`.
#[must_use]
pub fn fundamental_forms(
    f: &dyn Fn(f64, f64) -> Vec3,
    u: f64,
    v: f64,
    h: f64,
) -> FundamentalForms {
    assert!(h > 0.0, "fundamental_forms requires h > 0");
    let fu = (f(u + h, v) - f(u - h, v)) * (0.5 / h);
    let fv = (f(u, v + h) - f(u, v - h)) * (0.5 / h);
    let p = f(u, v);
    let fuu = (f(u + h, v) - p * 2.0 + f(u - h, v)) * (1.0 / (h * h));
    let fvv = (f(u, v + h) - p * 2.0 + f(u, v - h)) * (1.0 / (h * h));
    let fuv = (f(u + h, v + h) - f(u + h, v - h) - f(u - h, v + h) + f(u - h, v - h))
        * (0.25 / (h * h));
    let nvec = fu.cross(&fv);
    let nm = nvec.magnitude();
    assert!(nm > 0.0, "surface is degenerate at (u, v)");
    let n = nvec * (1.0 / nm);
    FundamentalForms {
        e: fu.dot(&fu),
        f: fu.dot(&fv),
        g: fv.dot(&fv),
        l: fuu.dot(&n),
        m: fuv.dot(&n),
        n: fvv.dot(&n),
    }
}

/// Gaussian curvature K = (LN − M²) / (EG − F²).
#[must_use]
pub fn gaussian_curvature(forms: &FundamentalForms) -> f64 {
    (forms.l * forms.n - forms.m * forms.m) / (forms.e * forms.g - forms.f * forms.f)
}

/// Mean curvature H = (EN − 2FM + GL) / (2 (EG − F²)).
#[must_use]
pub fn mean_curvature(forms: &FundamentalForms) -> f64 {
    (forms.e * forms.n - 2.0 * forms.f * forms.m + forms.g * forms.l)
        / (2.0 * (forms.e * forms.g - forms.f * forms.f))
}

/// Principal curvatures `(κ₁, κ₂)` with κ₁ ≥ κ₂:
/// H ± sqrt(H² − K).
#[must_use]
pub fn principal_curvatures(forms: &FundamentalForms) -> (f64, f64) {
    let h = mean_curvature(forms);
    let k = gaussian_curvature(forms);
    let d = (h * h - k).max(0.0).sqrt();
    (h + d, h - d)
}

/// Surface area of a parametric patch by the midpoint rule on
/// `nu` x `nv` cells: Σ |f_u × f_v| du dv, partials by central
/// differences at each cell midpoint.
///
/// # Panics
/// Panics unless `nu >= 1` and `nv >= 1`.
#[must_use]
pub fn surface_area_parametric(
    f: &dyn Fn(f64, f64) -> Vec3,
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
) -> f64 {
    assert!(nu >= 1 && nv >= 1, "surface_area_parametric requires nu, nv >= 1");
    let du = (u_range.1 - u_range.0) / nu as f64;
    let dv = (v_range.1 - v_range.0) / nv as f64;
    let mut area = 0.0;
    for i in 0..nu {
        let u = u_range.0 + (i as f64 + 0.5) * du;
        for j in 0..nv {
            let v = v_range.0 + (j as f64 + 0.5) * dv;
            let fu = (f(u + 0.5 * du, v) - f(u - 0.5 * du, v)) * (1.0 / du);
            let fv = (f(u, v + 0.5 * dv) - f(u, v - 0.5 * dv)) * (1.0 / dv);
            area += fu.cross(&fv).magnitude() * du * dv;
        }
    }
    area
}

/// Möbius strip of center radius `r` and half-width `w`:
/// u ∈ [0, 2π) around, v ∈ [−1, 1] across.
#[must_use]
pub fn mobius_strip(u: f64, v: f64, r: f64, w: f64) -> Vec3 {
    let q = r + w * v * (u / 2.0).cos();
    Vec3::new(q * u.cos(), w * v * (u / 2.0).sin(), q * u.sin())
}

/// Figure-8 immersion of the Klein bottle, u, v ∈ [0, 2π).
#[must_use]
pub fn klein_bottle(u: f64, v: f64) -> Vec3 {
    let r = 2.0;
    let a = r + (u / 2.0).cos() * v.sin() - (u / 2.0).sin() * (2.0 * v).sin();
    Vec3::new(
        a * u.cos(),
        a * u.sin(),
        (u / 2.0).sin() * v.sin() + (u / 2.0).cos() * (2.0 * v).sin(),
    )
}

/// Enneper's minimal surface.
#[must_use]
pub fn enneper(u: f64, v: f64) -> Vec3 {
    Vec3::new(
        u - u * u * u / 3.0 + u * v * v,
        v - v * v * v / 3.0 + v * u * u,
        u * u - v * v,
    )
}

/// Catenoid (minimal): u around, v along the axis, waist radius `c`.
#[must_use]
pub fn catenoid(u: f64, v: f64, c: f64) -> Vec3 {
    let r = c * (v / c).cosh();
    Vec3::new(r * u.cos(), v, r * u.sin())
}

/// Helicoid (minimal): pitch parameter `c`.
#[must_use]
pub fn helicoid(u: f64, v: f64, c: f64) -> Vec3 {
    Vec3::new(v * u.cos(), c * u, v * u.sin())
}

/// Monkey saddle z = u³ − 3uv².
#[must_use]
pub fn monkey_saddle(u: f64, v: f64) -> Vec3 {
    Vec3::new(u, v, u * u * u - 3.0 * u * v * v)
}

/// Dini's surface (constant negative curvature −1/(a² + b²)):
/// a twisted pseudosphere. v ∈ (0, π).
#[must_use]
pub fn dini(u: f64, v: f64, a: f64, b: f64) -> Vec3 {
    Vec3::new(
        a * u.cos() * v.sin(),
        a * (v.cos() + (v / 2.0).tan().ln()) + b * u,
        a * u.sin() * v.sin(),
    )
}

/// Boy's surface (Apéry parametrization of the real projective
/// plane), u ∈ [−π/2, π/2], v ∈ [0, π/2].
#[must_use]
pub fn boy_surface(u: f64, v: f64) -> Vec3 {
    let d = 2.0 - std::f64::consts::SQRT_2 * (3.0 * u).sin() * (2.0 * v).sin();
    let cv2 = v.cos() * v.cos();
    Vec3::new(
        (std::f64::consts::SQRT_2 * cv2 * (2.0 * u).cos() + u.cos() * (2.0 * v).sin()) / d,
        (std::f64::consts::SQRT_2 * cv2 * (2.0 * u).sin() - u.sin() * (2.0 * v).sin()) / d,
        3.0 * cv2 / d,
    )
}

fn sign_pow(x: f64, e: f64) -> f64 {
    x.signum() * x.abs().powf(e)
}

/// Superellipsoid with semi-axes `a` and exponents `e1` (latitude),
/// `e2` (longitude): u = longitude ∈ [−π, π], v = latitude ∈
/// [−π/2, π/2].
#[must_use]
pub fn superellipsoid(u: f64, v: f64, a: Vec3, e1: f64, e2: f64) -> Vec3 {
    Vec3::new(
        a.x * sign_pow(v.cos(), e1) * sign_pow(u.cos(), e2),
        a.y * sign_pow(v.sin(), e1),
        a.z * sign_pow(v.cos(), e1) * sign_pow(u.sin(), e2),
    )
}

/// Gielis superformula in the plane:
/// r(θ) = (|cos(mθ/4)/a|^n2 + |sin(mθ/4)/b|^n3)^(−1/n1).
///
/// # Panics
/// Panics unless `a, b > 0` and `n1 != 0`.
#[must_use]
pub fn supershape_2d(theta: f64, m: f64, n1: f64, n2: f64, n3: f64, a: f64, b: f64) -> Vec2 {
    assert!(a > 0.0 && b > 0.0 && n1 != 0.0, "supershape requires a, b > 0 and n1 != 0");
    let t1 = ((m * theta / 4.0).cos() / a).abs().powf(n2);
    let t2 = ((m * theta / 4.0).sin() / b).abs().powf(n3);
    let r = (t1 + t2).powf(-1.0 / n1);
    Vec2::new(r * theta.cos(), r * theta.sin())
}

/// 3-D supershape: the spherical product of two superformulas,
/// `params = [m1, n11, n12, n13, a1, b1, m2, n21, n22, n23, a2, b2]`
/// with θ = longitude ∈ [−π, π], φ = latitude ∈ [−π/2, π/2].
#[must_use]
pub fn supershape_3d(theta: f64, phi: f64, params: &[f64; 12]) -> Vec3 {
    let r1 = supershape_2d(theta, params[0], params[1], params[2], params[3], params[4], params[5])
        .magnitude();
    let r2 = supershape_2d(phi, params[6], params[7], params[8], params[9], params[10], params[11])
        .magnitude();
    Vec3::new(
        r1 * theta.cos() * r2 * phi.cos(),
        r2 * phi.sin(),
        r1 * theta.sin() * r2 * phi.cos(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezier_corners_and_subdivision() {
        let mut control = [[Vec3::ZERO; 4]; 4];
        for (i, row) in control.iter_mut().enumerate() {
            for (j, p) in row.iter_mut().enumerate() {
                *p = Vec3::new(i as f64, j as f64, ((i * j) as f64).sin());
            }
        }
        let patch = BezierPatch { control };
        assert!((patch.eval(0.0, 0.0) - control[0][0]).magnitude() < 1e-15);
        assert!((patch.eval(1.0, 0.0) - control[3][0]).magnitude() < 1e-15);
        assert!((patch.eval(0.0, 1.0) - control[0][3]).magnitude() < 1e-15);
        assert!((patch.eval(1.0, 1.0) - control[3][3]).magnitude() < 1e-15);
        // Subdivision reproduces the surface exactly.
        let quads = patch.subdivide();
        for (u, v) in [(0.3, 0.2), (0.7, 0.1), (0.2, 0.9), (0.8, 0.6), (0.5, 0.5)] {
            let exact = patch.eval(u, v);
            let (qi, lu) = if u <= 0.5 { (0, 2.0 * u) } else { (1, 2.0 * u - 1.0) };
            let (qj, lv) = if v <= 0.5 { (0, 2.0 * v) } else { (2, 2.0 * v - 1.0) };
            let sub = quads[qi + qj].eval(lu, lv);
            assert!((sub - exact).magnitude() < 1e-12);
        }
        // Derivatives match finite differences.
        let h = 1e-6;
        let du = (patch.eval(0.4 + h, 0.3) - patch.eval(0.4 - h, 0.3)) * (0.5 / h);
        assert!((patch.du(0.4, 0.3) - du).magnitude() < 1e-6);
        let n = patch.normal(0.4, 0.3);
        assert!((n.magnitude() - 1.0).abs() < 1e-12);
        assert!(n.dot(&patch.du(0.4, 0.3)).abs() < 1e-9);
    }

    #[test]
    fn test_bspline_clamped_interpolates_corners() {
        let control: Vec<Vec<Vec3>> = (0..5)
            .map(|i| {
                (0..4)
                    .map(|j| Vec3::new(i as f64, j as f64, (i + j) as f64 * 0.1))
                    .collect()
            })
            .collect();
        let s = BSplineSurface::uniform(3, 2, control.clone());
        assert!((s.eval(0.0, 0.0) - control[0][0]).magnitude() < 1e-12);
        assert!((s.eval(1.0, 1.0) - control[4][3]).magnitude() < 1e-12);
        // Convex hull: y stays within the control range.
        for &(u, v) in &[(0.2, 0.7), (0.5, 0.5), (0.9, 0.1)] {
            let p = s.eval(u, v);
            assert!(p.y >= 0.0 && p.y <= 3.0);
        }
        let n = s.normal(0.5, 0.5);
        assert!((n.magnitude() - 1.0).abs() < 1e-9);
        let m = s.to_mesh(8, 8);
        assert_eq!(m.vertices.len(), 81);
    }

    #[test]
    fn test_nurbs_sphere_exact_and_torus() {
        let r = 1.7;
        let s = NurbsSurface::sphere(r);
        for i in 0..=20 {
            for j in 0..=20 {
                let p = s.eval(i as f64 / 20.0, j as f64 / 20.0);
                assert!(
                    (p.magnitude() - r).abs() < 1e-12,
                    "NURBS sphere point off surface: {}",
                    p.magnitude()
                );
            }
        }
        let t = NurbsSurface::torus(2.0, 0.5);
        for i in 0..=15 {
            for j in 0..=15 {
                let p = t.eval(i as f64 / 15.0, j as f64 / 15.0);
                let radial = (p.x * p.x + p.z * p.z).sqrt();
                let d = ((radial - 2.0).powi(2) + p.y * p.y).sqrt();
                assert!((d - 0.5).abs() < 1e-12, "NURBS torus point off surface");
            }
        }
        let c = NurbsSurface::cylinder(1.0, 3.0);
        for i in 0..=15 {
            let p = c.eval(i as f64 / 15.0, 0.4);
            assert!(((p.x * p.x + p.z * p.z).sqrt() - 1.0).abs() < 1e-12);
            assert!((p.y - 1.2).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bezier_patch_to_mesh_grid_and_corner_interpolation() {
        let mut control = [[Vec3::ZERO; 4]; 4];
        for (i, row) in control.iter_mut().enumerate() {
            for (j, p) in row.iter_mut().enumerate() {
                *p = Vec3::new(i as f64, j as f64, (i as f64 - 1.5) * (j as f64 - 1.5));
            }
        }
        let patch = BezierPatch { control };
        for &(nu, nv) in &[(1usize, 1usize), (4, 6), (12, 12)] {
            let m = patch.to_mesh(nu, nv);
            // Open in both directions: (nu+1)·(nv+1) vertices and
            // 2·nu·nv triangles.
            assert_eq!(m.vertices.len(), (nu + 1) * (nv + 1), "{nu}x{nv} vertices");
            assert_eq!(m.indices.len(), 2 * nu * nv, "{nu}x{nv} triangles");
            // Row-major (i along u, j along v) sampling of eval.
            for i in 0..=nu {
                for j in 0..=nv {
                    let u = i as f64 / nu as f64;
                    let v = j as f64 / nv as f64;
                    let got = m.vertices[i * (nv + 1) + j];
                    assert!(
                        (got - patch.eval(u, v)).magnitude() < 1e-15,
                        "vertex ({i}, {j})"
                    );
                }
            }
            // Endpoint interpolation: the four mesh corners are the
            // four corner control points, exactly.
            assert!((m.vertices[0] - control[0][0]).magnitude() < 1e-15);
            assert!((m.vertices[nv] - control[0][3]).magnitude() < 1e-15);
            assert!(
                (m.vertices[nu * (nv + 1)] - control[3][0]).magnitude() < 1e-15
            );
            assert!(
                (m.vertices[nu * (nv + 1) + nv] - control[3][3]).magnitude() < 1e-15
            );
            // Convex hull property: every vertex lies inside the
            // control net's bounding box.
            let (mut lo, mut hi) = (Vec3::ZERO, Vec3::ZERO);
            for (i, row) in control.iter().enumerate() {
                for (j, p) in row.iter().enumerate() {
                    if i == 0 && j == 0 {
                        lo = *p;
                        hi = *p;
                    }
                    lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
                    hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
                }
            }
            for v in &m.vertices {
                assert!(v.x >= lo.x - 1e-12 && v.x <= hi.x + 1e-12);
                assert!(v.y >= lo.y - 1e-12 && v.y <= hi.y + 1e-12);
                assert!(v.z >= lo.z - 1e-12 && v.z <= hi.z + 1e-12);
            }
        }
        // A planar patch tessellates to exactly its own area: control
        // points on the z = 0 plane spanning [0, 3]².
        let mut flat = [[Vec3::ZERO; 4]; 4];
        for (i, row) in flat.iter_mut().enumerate() {
            for (j, p) in row.iter_mut().enumerate() {
                *p = Vec3::new(i as f64, j as f64, 0.0);
            }
        }
        let plane = BezierPatch { control: flat };
        let m = plane.to_mesh(5, 7);
        assert!((m.surface_area() - 9.0).abs() < 1e-12, "flat patch area");
        // Refining the curved patch converges: the tessellated area
        // approaches the analytic parametric integral from below.
        let exact = surface_area_parametric(
            &|u, v| patch.eval(u, v),
            (0.0, 1.0),
            (0.0, 1.0),
            256,
            256,
        );
        let coarse = patch.to_mesh(4, 4).surface_area();
        let fine = patch.to_mesh(64, 64).surface_area();
        assert!((coarse - exact).abs() / exact < 0.05, "coarse {coarse} vs {exact}");
        assert!((fine - exact).abs() / exact < 1e-3, "fine {fine} vs {exact}");
        assert!(
            (fine - exact).abs() < (coarse - exact).abs(),
            "refinement converges: {coarse} then {fine} toward {exact}"
        );
    }

    #[test]
    fn test_nurbs_to_mesh_lands_exactly_on_the_exact_surfaces() {
        // The NURBS sphere is exact, so every tessellation vertex must
        // sit on the sphere to machine precision.
        let r = 1.7;
        let s = NurbsSurface::sphere(r);
        for &(nu, nv) in &[(8usize, 6usize), (24, 16)] {
            let m = s.to_mesh(nu, nv);
            assert_eq!(m.vertices.len(), (nu + 1) * (nv + 1));
            assert_eq!(m.indices.len(), 2 * nu * nv);
            for v in &m.vertices {
                assert!(
                    (v.magnitude() - r).abs() < 1e-12,
                    "sphere vertex at radius {}",
                    v.magnitude()
                );
            }
            // Clamped knots: the mesh corners interpolate the corner
            // control points, which for the sphere are the poles.
            assert!((m.vertices[0] - s.control[0][0]).magnitude() < 1e-12);
            assert!((m.vertices[nv] - s.control[0][4]).magnitude() < 1e-12);
            let last_row = nu * (nv + 1);
            assert!((m.vertices[last_row] - s.control[8][0]).magnitude() < 1e-12);
            assert!((m.vertices[last_row + nv] - s.control[8][4]).magnitude() < 1e-12);
            assert!((m.vertices[0] - Vec3::new(0.0, -r, 0.0)).magnitude() < 1e-12);
            assert!((m.vertices[nv] - Vec3::new(0.0, r, 0.0)).magnitude() < 1e-12);
            // Grid vertices reproduce eval on the whole domain.
            for i in 0..=nu {
                for j in 0..=nv {
                    let (u, v) = (i as f64 / nu as f64, j as f64 / nv as f64);
                    assert!(
                        (m.vertices[i * (nv + 1) + j] - s.eval(u, v)).magnitude() < 1e-12
                    );
                }
            }
        }
        // Refining converges to 4πr² from below (inscribed polyhedron).
        let exact = 4.0 * std::f64::consts::PI * r * r;
        let coarse = s.to_mesh(16, 12).surface_area();
        let fine = s.to_mesh(96, 64).surface_area();
        assert!(coarse < fine, "refinement increases the inscribed area");
        assert!(fine < exact + 1e-9, "inscribed area below the exact one");
        assert!((fine - exact).abs() / exact < 2e-3, "{fine} vs {exact}");

        // The NURBS cylinder: every vertex at radius r, heights spanning
        // the full [0, h] and matching the v parameter linearly.
        let (rc, h) = (1.0, 3.0);
        let c = NurbsSurface::cylinder(rc, h);
        let m = c.to_mesh(16, 4);
        assert_eq!(m.vertices.len(), 17 * 5);
        for (k, v) in m.vertices.iter().enumerate() {
            let radial = (v.x * v.x + v.z * v.z).sqrt();
            assert!((radial - rc).abs() < 1e-12, "cylinder radius {radial}");
            let j = k % 5;
            assert!(
                (v.y - h * j as f64 / 4.0).abs() < 1e-12,
                "cylinder height {} at row {j}",
                v.y
            );
        }
        // Lateral area 2πrh, approached from below by the inscribed
        // 16-sided prism and recovered to <1e-4 when refined.
        let lateral = 2.0 * std::f64::consts::PI * rc * h;
        let a16 = m.surface_area();
        assert!(a16 < lateral, "inscribed prism area {a16} exceeds {lateral}");
        assert!((lateral - a16) / lateral < 0.01, "16 sides: {a16} vs {lateral}");
        let a256 = c.to_mesh(256, 4).surface_area();
        assert!(a256 < lateral);
        assert!((lateral - a256) / lateral < 1e-4, "256 sides: {a256} vs {lateral}");

        // The NURBS torus: every vertex on the tube.
        let t = NurbsSurface::torus(2.0, 0.5);
        for v in &t.to_mesh(20, 12).vertices {
            let radial = (v.x * v.x + v.z * v.z).sqrt();
            let d = ((radial - 2.0).powi(2) + v.y * v.y).sqrt();
            assert!((d - 0.5).abs() < 1e-12, "torus vertex off the tube: {d}");
        }
    }

    #[test]
    fn test_curvatures_of_classic_surfaces() {
        let r = 2.0;
        let sphere = |u: f64, v: f64| {
            Vec3::new(r * v.sin() * u.cos(), r * v.cos(), r * v.sin() * u.sin())
        };
        for &(u, v) in &[(0.3, 0.8), (1.2, 1.5), (4.0, 2.0)] {
            let ff = fundamental_forms(&sphere, u, v, 1e-4);
            assert!((gaussian_curvature(&ff) - 1.0 / (r * r)).abs() < 1e-4);
            let (k1, k2) = principal_curvatures(&ff);
            assert!((k1.abs() - 1.0 / r).abs() < 1e-4);
            assert!((k2.abs() - 1.0 / r).abs() < 1e-4);
        }
        // Minimal surfaces: mean curvature 0.
        for &(u, v) in &[(0.5, 0.3), (2.0, -0.6), (4.2, 1.0)] {
            let cat = |u: f64, v: f64| catenoid(u, v, 1.0);
            assert!(mean_curvature(&fundamental_forms(&cat, u, v, 1e-4)).abs() < 1e-4);
            let hel = |u: f64, v: f64| helicoid(u, v, 0.8);
            assert!(mean_curvature(&fundamental_forms(&hel, u, v, 1e-4)).abs() < 1e-4);
            let enn = |u: f64, v: f64| enneper(u, v);
            assert!(mean_curvature(&fundamental_forms(&enn, u, v, 1e-4)).abs() < 1e-4);
        }
        // Dini: constant curvature -1/(a^2+b^2).
        let (a, b) = (1.0, 0.2);
        for &(u, v) in &[(0.5, 1.0), (2.0, 1.8), (5.0, 0.7)] {
            let d = |u: f64, v: f64| dini(u, v, a, b);
            let k = gaussian_curvature(&fundamental_forms(&d, u, v, 1e-4));
            assert!((k + 1.0 / (a * a + b * b)).abs() < 1e-3, "dini curvature {k}");
        }
    }

    #[test]
    fn test_surface_area_sphere_and_ruled() {
        let r = 1.5;
        let sphere = |u: f64, v: f64| {
            Vec3::new(r * v.sin() * u.cos(), r * v.cos(), r * v.sin() * u.sin())
        };
        let area = surface_area_parametric(
            &sphere,
            (0.0, 2.0 * std::f64::consts::PI),
            (0.0, std::f64::consts::PI),
            64,
            64,
        );
        let exact = 4.0 * std::f64::consts::PI * r * r;
        assert!((area - exact).abs() / exact < 1e-3);
        // A ruled surface between parallel unit segments is a unit
        // square.
        let c1 = |u: f64| Vec3::new(u, 0.0, 0.0);
        let c2 = |u: f64| Vec3::new(u, 1.0, 0.0);
        let ruled = |u: f64, v: f64| ruled_surface(&c1, &c2, u, v);
        let area = surface_area_parametric(&ruled, (0.0, 1.0), (0.0, 1.0), 8, 8);
        assert!((area - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_coons_patch_interpolates_boundaries() {
        let c0 = |u: f64| Vec3::new(u, 0.0, (std::f64::consts::PI * u).sin());
        let c1 = |u: f64| Vec3::new(u, 1.0, 0.3 * u);
        let d0 = |v: f64| c0(0.0) * (1.0 - v * v) + c1(0.0) * (v * v);
        let d1 = |v: f64| c0(1.0) * (1.0 - v.sqrt()) + c1(1.0) * v.sqrt();
        for k in 0..=10 {
            let t = k as f64 / 10.0;
            assert!((coons_patch(&c0, &c1, &d0, &d1, t, 0.0) - c0(t)).magnitude() < 1e-12);
            assert!((coons_patch(&c0, &c1, &d0, &d1, t, 1.0) - c1(t)).magnitude() < 1e-12);
            assert!((coons_patch(&c0, &c1, &d0, &d1, 0.0, t) - d0(t)).magnitude() < 1e-12);
            assert!((coons_patch(&c0, &c1, &d0, &d1, 1.0, t) - d1(t)).magnitude() < 1e-12);
        }
    }

    #[test]
    fn test_named_surfaces_basics() {
        // Möbius: center circle at radius r.
        let p = mobius_strip(1.0, 0.0, 2.0, 0.5);
        assert!((p.magnitude() - 2.0).abs() < 1e-12);
        // Superellipsoid with e1 = e2 = 1 is an ellipsoid.
        let a = Vec3::new(1.0, 2.0, 3.0);
        for &(u, v) in &[(0.3, 0.2), (2.0, -0.9), (-1.0, 1.2)] {
            let p = superellipsoid(u, v, a, 1.0, 1.0);
            let e = (p.x / a.x).powi(2) + (p.y / a.y).powi(2) + (p.z / a.z).powi(2);
            assert!((e - 1.0).abs() < 1e-12);
        }
        // Supershape with m = 0 and unit exponents is a circle.
        for k in 0..8 {
            let th = k as f64 * 0.7;
            let p = supershape_2d(th, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0);
            assert!((p.magnitude() - 1.0).abs() < 1e-12);
        }
        // 3-D supershape with circles gives the unit sphere.
        let params = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        for &(t, p) in &[(0.4, 0.2), (2.0, -1.0), (-2.5, 1.3)] {
            let q = supershape_3d(t, p, &params);
            assert!((q.magnitude() - 1.0).abs() < 1e-12);
        }
        // Monkey saddle and Enneper pass through the origin.
        assert!(monkey_saddle(0.0, 0.0).magnitude() < 1e-15);
        assert!(enneper(0.0, 0.0).magnitude() < 1e-15);
        // Klein bottle and Boy surface produce finite points.
        for &(u, v) in &[(0.1, 0.2), (3.0, 4.0), (1.5, 0.8)] {
            assert!(klein_bottle(u, v).magnitude().is_finite());
            assert!(boy_surface(u, v).magnitude().is_finite());
        }
        // Revolution of a circle profile hits the sphere.
        let prof = |t: f64| Vec2::new(t.sin(), -t.cos());
        let q = surface_of_revolution(&prof, 1.0, 2.0);
        assert!((q.magnitude() - 1.0).abs() < 1e-12);
    }
}
