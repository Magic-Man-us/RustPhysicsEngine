//! Triangular finite elements in the plane.
//!
//! # The linear triangle
//!
//! On a triangle the three barycentric coordinates are themselves the
//! linear shape functions, and their gradients are constant. That single
//! fact does most of the work: the stiffness integral
//! `integral grad(phi_i) . grad(phi_j)` has a constant integrand, so it is
//! the gradient product times the triangle's area, with no quadrature
//! involved and no error introduced. The whole element matrix for the
//! Laplacian is
//!
//! ```text
//! K_ij = (b_i b_j + c_i c_j) / (4 A)
//! ```
//!
//! where `b` and `c` are the edge-opposite coordinate differences and `A`
//! is the signed area. The two-dimensional method inherits everything the
//! one-dimensional one has -- Galerkin orthogonality, energy
//! minimisation, best approximation in the energy norm -- because none of
//! those arguments mentions the dimension.
//!
//! # What the mesh has to guarantee
//!
//! Two conditions matter and they are different in kind.
//!
//! *Conformity* is structural: two triangles meet along a whole shared
//! edge or at a single shared vertex, never at a vertex hanging in the
//! middle of a neighbour's edge. Without it the assembled function is not
//! continuous and the space is not a subspace of `H1`, so the theory does
//! not apply at all. It is checked here by counting: every edge belongs
//! to one triangle or two, never more.
//!
//! *Shape* is quantitative. The interpolation error carries a factor of
//! `1/sin(theta_min)`, so a mesh of slivers converges at the same rate
//! with a much worse constant. [`FemMesh2::quality_min_angle`] reports
//! the worst angle in the mesh, and uniform refinement leaves it exactly
//! unchanged -- the four children of a triangle are all similar to their
//! parent, which is the property that makes repeated refinement safe and
//! that a red-green or longest-edge scheme has to work to recover.
//!
//! # Delaunay and the maximum principle
//!
//! The off-diagonal stiffness entry for an interior edge is
//! `-(cot alpha + cot beta)/2`, the two angles opposite the edge in the
//! triangles sharing it. It is nonpositive exactly when those angles sum
//! to no more than `pi` -- which is the Delaunay condition. So a Delaunay
//! triangulation gives an M-matrix, and an M-matrix gives a discrete
//! maximum principle: a nonnegative load produces a nonnegative solution,
//! and a harmonic one attains its extremes on the boundary. On a badly
//! shaped non-Delaunay mesh the discrete solution can overshoot its own
//! boundary data while still converging, which is exactly the kind of
//! defect a plausibility check on a picture would miss.

use crate::error::{GeomError, SolveError};
use crate::linalg::matrix::Matrix;
use crate::linalg::sparse::{pcg_jacobi, CsrMatrix};
use crate::math::Vec2;

/// A conforming triangulation of a planar region.
#[derive(Debug, Clone, PartialEq)]
pub struct FemMesh2 {
    /// Vertex coordinates.
    pub nodes: Vec<Vec2>,
    /// Triangles as node index triples, counterclockwise.
    pub tris: Vec<[usize; 3]>,
    /// Indices of the nodes lying on the boundary, ascending.
    pub boundary: Vec<usize>,
}

/// An undirected edge as an ordered index pair.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

impl FemMesh2 {
    /// Builds a mesh from nodes and triangles, orienting every triangle
    /// counterclockwise and deriving the boundary from the edge counts.
    ///
    /// Orienting rather than rejecting is deliberate: a triangle listed
    /// clockwise describes the same element, and the sign of its area is
    /// a labelling convention rather than a property of the geometry. A
    /// *zero* area is not, and is refused.
    ///
    /// # Errors
    ///
    /// [`GeomError::Empty`] with no triangles;
    /// [`GeomError::InvalidArgument`] for an out-of-range index or a
    /// repeated vertex within one triangle; [`GeomError::Degenerate`] for
    /// a zero-area triangle; [`GeomError::NotManifold`] if any edge is
    /// shared by more than two triangles.
    pub fn new(nodes: Vec<Vec2>, tris: Vec<[usize; 3]>) -> Result<Self, GeomError> {
        if tris.is_empty() || nodes.is_empty() {
            return Err(GeomError::Empty);
        }
        if nodes.iter().any(|p| !(p.x.is_finite() && p.y.is_finite())) {
            return Err(GeomError::InvalidArgument("node coordinates must be finite"));
        }
        let mut oriented = Vec::with_capacity(tris.len());
        for t in &tris {
            if t.iter().any(|&i| i >= nodes.len()) {
                return Err(GeomError::InvalidArgument("triangle index out of range"));
            }
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                return Err(GeomError::InvalidArgument("triangle repeats a vertex"));
            }
            let area = signed_area(&nodes, t);
            if area == 0.0 {
                return Err(GeomError::Degenerate("zero-area triangle"));
            }
            oriented.push(if area > 0.0 { *t } else { [t[0], t[2], t[1]] });
        }
        // An edge in one triangle is a boundary edge, in two an interior
        // one, and in three or more the surface is not a surface.
        let mut counts: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for t in &oriented {
            for k in 0..3 {
                *counts.entry(edge_key(t[k], t[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        if counts.values().any(|&c| c > 2) {
            return Err(GeomError::NotManifold);
        }
        let mut on_boundary = vec![false; nodes.len()];
        for (&(a, b), &c) in &counts {
            if c == 1 {
                on_boundary[a] = true;
                on_boundary[b] = true;
            }
        }
        let boundary = (0..nodes.len()).filter(|&i| on_boundary[i]).collect();
        Ok(Self { nodes, tris: oriented, boundary })
    }

    /// A right-triangle mesh of the rectangle `[0, w] x [0, h]`, each
    /// cell split along one diagonal.
    ///
    /// The diagonals all run the same way, which makes the mesh Delaunay
    /// -- every triangle is right-angled, so no angle opposite an edge
    /// exceeds a right angle and the pair opposite any interior edge sums
    /// to `pi` exactly.
    ///
    /// # Errors
    ///
    /// [`GeomError::InvalidArgument`] for a non-positive extent or a zero
    /// subdivision count.
    pub fn rect(w: f64, h: f64, nx: usize, ny: usize) -> Result<Self, GeomError> {
        if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
            return Err(GeomError::InvalidArgument("rectangle extents must be positive"));
        }
        if nx == 0 || ny == 0 {
            return Err(GeomError::InvalidArgument("need at least one cell in each direction"));
        }
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push(Vec2::new(w * i as f64 / nx as f64, h * j as f64 / ny as f64));
            }
        }
        let at = |i: usize, j: usize| j * (nx + 1) + i;
        let mut tris = Vec::with_capacity(2 * nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                tris.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                tris.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        Self::new(nodes, tris)
    }

    /// A fan-and-rings mesh of the disk of radius `r`, with `n` rings.
    ///
    /// The rings carry `6k` points at radius `k r / n`, which keeps the
    /// arc spacing roughly equal to the radial spacing and so keeps the
    /// triangles from degenerating towards the rim -- a fixed point count
    /// per ring would make the outer triangles long and thin.
    ///
    /// # Errors
    ///
    /// [`GeomError::InvalidArgument`] for a non-positive radius or fewer
    /// than one ring.
    pub fn disk(r: f64, n: usize) -> Result<Self, GeomError> {
        if !r.is_finite() || r <= 0.0 {
            return Err(GeomError::InvalidArgument("disk radius must be positive"));
        }
        if n == 0 {
            return Err(GeomError::InvalidArgument("need at least one ring"));
        }
        let tau = std::f64::consts::TAU;
        let mut nodes = vec![Vec2::ZERO];
        let mut ring_start = vec![0usize];
        for k in 1..=n {
            ring_start.push(nodes.len());
            let count = 6 * k;
            let radius = r * k as f64 / n as f64;
            for m in 0..count {
                let a = tau * m as f64 / count as f64;
                nodes.push(Vec2::new(radius * a.cos(), radius * a.sin()));
            }
        }
        ring_start.push(nodes.len());
        let mut tris = Vec::new();
        // Innermost ring: a fan from the centre.
        for m in 0..6 {
            tris.push([0, ring_start[1] + m, ring_start[1] + (m + 1) % 6]);
        }
        // Between consecutive rings the counts differ by six, so walk
        // both rings by angle and emit whichever triangle advances the
        // one that is behind. This is the same merge that keeps a
        // triangle strip between two unequal polylines conforming.
        for k in 1..n {
            let (inner, outer) = (ring_start[k], ring_start[k + 1]);
            let (ni, no) = (6 * k, 6 * (k + 1));
            let (mut i, mut o) = (0usize, 0usize);
            while i < ni || o < no {
                let ai = i as f64 / ni as f64;
                let ao = o as f64 / no as f64;
                if o >= no || (i < ni && ai <= ao) {
                    tris.push([inner + i % ni, outer + o % no, inner + (i + 1) % ni]);
                    i += 1;
                } else {
                    tris.push([inner + i % ni, outer + o % no, outer + (o + 1) % no]);
                    o += 1;
                }
            }
        }
        Self::new(nodes, tris)
    }

    /// A Delaunay triangulation of a point set.
    ///
    /// # Errors
    ///
    /// [`GeomError::Empty`] for fewer than three points, and whatever
    /// [`FemMesh2::new`] reports for a degenerate result -- collinear
    /// points produce no triangles at all.
    pub fn from_delaunay(points: &[Vec2]) -> Result<Self, GeomError> {
        if points.len() < 3 {
            return Err(GeomError::Empty);
        }
        let raw: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();
        let tris = crate::geometry::delaunay::delaunay_2d(&raw);
        if tris.is_empty() {
            return Err(GeomError::Degenerate("no triangles: the points may be collinear"));
        }
        Self::new(points.to_vec(), tris)
    }

    /// Splits every triangle into four by joining its edge midpoints.
    ///
    /// All four children are similar to the parent, so the mesh quality
    /// is preserved exactly rather than approximately: repeated
    /// refinement of a good mesh stays good, and repeated refinement of a
    /// sliver never recovers.
    pub fn refine_uniform(&self) -> Self {
        let mut nodes = self.nodes.clone();
        let mut midpoint: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut tris = Vec::with_capacity(4 * self.tris.len());
        for t in &self.tris {
            let mut mid = [0usize; 3];
            for k in 0..3 {
                // Edge k joins vertices k+1 and k+2, so mid[k] is the
                // node opposite vertex k in the child layout below.
                let (a, b) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                let key = edge_key(a, b);
                mid[k] = *midpoint.entry(key).or_insert_with(|| {
                    nodes.push(Vec2::new(
                        0.5 * (self.nodes[a].x + self.nodes[b].x),
                        0.5 * (self.nodes[a].y + self.nodes[b].y),
                    ));
                    nodes.len() - 1
                });
            }
            tris.push([t[0], mid[2], mid[1]]);
            tris.push([mid[2], t[1], mid[0]]);
            tris.push([mid[1], mid[0], t[2]]);
            tris.push([mid[0], mid[1], mid[2]]);
        }
        // The children of a conforming mesh are conforming, so this
        // cannot fail; the boundary is rederived from the edge counts.
        Self::new(nodes, tris).expect("uniform refinement preserves conformity")
    }

    /// The smallest interior angle anywhere in the mesh, in radians.
    ///
    /// The interpolation error constant grows as `1/sin` of this, which
    /// is why it is the number to watch rather than the aspect ratio.
    pub fn quality_min_angle(&self) -> f64 {
        let mut worst = std::f64::consts::PI;
        for t in &self.tris {
            let p = [self.nodes[t[0]], self.nodes[t[1]], self.nodes[t[2]]];
            for k in 0..3 {
                let a = p[(k + 1) % 3] - p[k];
                let b = p[(k + 2) % 3] - p[k];
                // atan2 of the cross and dot rather than acos of the
                // normalised dot: the latter loses its precision exactly
                // where the answer matters, at a very small angle.
                let angle = (a.x * b.y - a.y * b.x).abs().atan2(a.x * b.x + a.y * b.y);
                worst = worst.min(angle);
            }
        }
        worst
    }

    /// The total area of the triangles.
    pub fn area(&self) -> f64 {
        self.tris.iter().map(|t| signed_area(&self.nodes, t)).sum()
    }

    /// The number of distinct edges, which the Euler characteristic
    /// relates to the node and triangle counts.
    pub fn edge_count(&self) -> usize {
        let mut set = std::collections::HashSet::new();
        for t in &self.tris {
            for k in 0..3 {
                set.insert(edge_key(t[k], t[(k + 1) % 3]));
            }
        }
        set.len()
    }
}

/// Twice-signed area over two: positive for a counterclockwise triple.
fn signed_area(nodes: &[Vec2], t: &[usize; 3]) -> f64 {
    let (a, b, c) = (nodes[t[0]], nodes[t[1]], nodes[t[2]]);
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
}

/// The constant gradients of the three linear shape functions on a
/// triangle, together with its area.
fn shape_gradients(nodes: &[Vec2], t: &[usize; 3]) -> ([Vec2; 3], f64) {
    let (a, b, c) = (nodes[t[0]], nodes[t[1]], nodes[t[2]]);
    let area = signed_area(nodes, t);
    let two = 2.0 * area;
    // grad(phi_i) is the inward normal of the opposite edge over twice
    // the area, which is what the barycentric coordinates differentiate
    // to.
    let g = [
        Vec2::new((b.y - c.y) / two, (c.x - b.x) / two),
        Vec2::new((c.y - a.y) / two, (a.x - c.x) / two),
        Vec2::new((a.y - b.y) / two, (b.x - a.x) / two),
    ];
    (g, area)
}

/// The centroid of a triangle.
fn centroid(nodes: &[Vec2], t: &[usize; 3]) -> Vec2 {
    let (a, b, c) = (nodes[t[0]], nodes[t[1]], nodes[t[2]]);
    Vec2::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0)
}

/// Assembled global matrices for the linear triangle.
///
/// `stiffness` is the Laplacian, `mass` is the consistent mass matrix,
/// and `load` is the source vector. Held as dense triplet lists before
/// compression, which is the cheap way to accumulate element
/// contributions that overlap.
struct Assembly {
    entries: Vec<(usize, usize, f64)>,
    load: Vec<f64>,
}

/// Assembles `-div(grad u) + reaction * u` against a source, both
/// evaluated at the element centroid.
///
/// Sampling the coefficients at the centroid is a one-point rule, exact
/// for a linear integrand and second-order otherwise -- the same order as
/// the element itself, so it costs nothing asymptotically. The stiffness
/// term needs no rule at all: the gradients are constant.
fn assemble(
    mesh: &FemMesh2,
    reaction: &dyn Fn(Vec2) -> f64,
    source: &dyn Fn(Vec2) -> f64,
) -> Result<Assembly, SolveError> {
    let n = mesh.nodes.len();
    let mut entries = Vec::with_capacity(9 * mesh.tris.len());
    let mut load = vec![0.0; n];
    for t in &mesh.tris {
        let (g, area) = shape_gradients(&mesh.nodes, t);
        let mid = centroid(&mesh.nodes, t);
        let r = reaction(mid);
        let f = source(mid);
        if !(r.is_finite() && f.is_finite()) {
            return Err(SolveError::InvalidArgument("coefficients must be finite"));
        }
        for j in 0..3 {
            // The one-point rule spreads the load equally over the three
            // vertices, since each shape function integrates to A/3.
            load[t[j]] += f * area / 3.0;
            for k in 0..3 {
                // The consistent mass matrix of a linear triangle is
                // A/12 off the diagonal and A/6 on it.
                let m = if j == k { area / 6.0 } else { area / 12.0 };
                entries.push((t[j], t[k], area * g[j].dot(&g[k]) + r * m));
            }
        }
    }
    Ok(Assembly { entries, load })
}

/// Applies Dirichlet data symmetrically: the known value is moved to the
/// right-hand side of every equation that saw it, and its own row and
/// column are replaced by a multiple of the identity.
///
/// A *multiple*, not the identity itself. Writing a bare one on the
/// diagonal is the textbook recipe and it is wrong for any problem whose
/// natural scale is not one: an elasticity matrix has diagonal entries of
/// order Young's modulus, so a row of one alongside rows of `1e10` gives
/// the assembled system a condition number of `1e10` that the physics
/// never had, and an iterative solver then delivers ten digits fewer than
/// it should. Scaling the pinned rows to match the rest costs nothing --
/// the solution is unchanged, since the row still says `u_i = g` -- and
/// removes the whole artefact.
fn apply_dirichlet(
    n: usize,
    entries: &[(usize, usize, f64)],
    load: &[f64],
    fixed: &[Option<f64>],
) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    let mut rhs = load.to_vec();
    for &(i, j, v) in entries {
        if let Some(g) = fixed[j] {
            if fixed[i].is_none() {
                rhs[i] -= v * g;
            }
        }
    }
    // A representative diagonal magnitude of the free part of the
    // system, accumulated before the duplicate triplets are merged.
    let mut diagonal = vec![0.0; n];
    for &(i, j, v) in entries {
        if i == j {
            diagonal[i] += v;
        }
    }
    let free: Vec<f64> = (0..n).filter(|&i| fixed[i].is_none()).map(|i| diagonal[i].abs()).collect();
    let pivot = if free.is_empty() {
        1.0
    } else {
        free.iter().sum::<f64>() / free.len() as f64
    };
    let pivot = if pivot > 0.0 { pivot } else { 1.0 };
    let mut kept: Vec<(usize, usize, f64)> = entries
        .iter()
        .copied()
        .filter(|&(i, j, _)| fixed[i].is_none() && fixed[j].is_none())
        .collect();
    for (i, slot) in fixed.iter().enumerate().take(n) {
        if let Some(g) = *slot {
            kept.push((i, i, pivot));
            rhs[i] = pivot * g;
        }
    }
    (kept, rhs)
}

/// Solves `-div(grad u) = f` on the mesh with the given Dirichlet data.
///
/// `dirichlet` is consulted at every boundary node; returning `None`
/// leaves that node free, which imposes the natural zero-flux condition
/// there. Returning `None` everywhere leaves the constant in the kernel
/// and is reported as [`SolveError::Singular`].
///
/// The system is symmetric positive definite once the data is applied, so
/// it is solved by Jacobi-preconditioned conjugate gradients.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for non-finite data,
/// [`SolveError::Singular`] if nothing pins the solution, and
/// [`SolveError::NoConvergence`] if the iteration stalls.
pub fn fem_2d_poisson(
    mesh: &FemMesh2,
    f: &dyn Fn(Vec2) -> f64,
    dirichlet: &dyn Fn(Vec2) -> Option<f64>,
) -> Result<Vec<f64>, SolveError> {
    fem_2d_reaction_diffusion(mesh, &|_| 0.0, f, dirichlet)
}

/// Solves `-div(grad u) + c u = f` on the mesh with Dirichlet data.
///
/// A positive `c` is a reaction term and keeps the problem coercive; a
/// negative one is the Helmholtz operator `-lap - k^2`, which loses
/// positive definiteness once `k^2` passes the first eigenvalue of the
/// domain. See [`fem_2d_helmholtz`] for that case, which needs a
/// different solver.
///
/// # Errors
///
/// As [`fem_2d_poisson`], and [`SolveError::NotPositiveDefinite`] if the
/// reaction term makes the system indefinite.
pub fn fem_2d_reaction_diffusion(
    mesh: &FemMesh2,
    c: &dyn Fn(Vec2) -> f64,
    f: &dyn Fn(Vec2) -> f64,
    dirichlet: &dyn Fn(Vec2) -> Option<f64>,
) -> Result<Vec<f64>, SolveError> {
    let n = mesh.nodes.len();
    let asm = assemble(mesh, c, f)?;
    let mut fixed = vec![None; n];
    let mut any = false;
    for &b in &mesh.boundary {
        if let Some(g) = dirichlet(mesh.nodes[b]) {
            if !g.is_finite() {
                return Err(SolveError::InvalidArgument("boundary data must be finite"));
            }
            fixed[b] = Some(g);
            any = true;
        }
    }
    if !any {
        // Every row of the pure Neumann Laplacian sums to zero because
        // the shape functions sum to one, so their gradients sum to
        // zero. A reaction term breaks that and pins the solution.
        let mut row_sum = vec![0.0; n];
        for &(i, _, v) in &asm.entries {
            row_sum[i] += v;
        }
        let scale = asm
            .entries
            .iter()
            .filter(|(i, j, _)| i == j)
            .map(|(_, _, v)| v.abs())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        if row_sum.iter().all(|s| s.abs() <= 1e-12 * scale) {
            return Err(SolveError::Singular);
        }
    }
    let (entries, rhs) = apply_dirichlet(n, &asm.entries, &asm.load, &fixed);
    let matrix = CsrMatrix::from_triplets(n, n, &entries);
    // The tolerance pcg_jacobi takes is relative to the norm of the
    // right-hand side, so it is passed as a pure number. Scaling it by
    // the data would loosen it by however many orders of magnitude the
    // data happens to span.
    pcg_jacobi(&matrix, &rhs, 1e-14, 20 * n + 500)
}

/// The assembled stiffness matrix of the Laplacian, with no boundary
/// conditions applied.
///
/// The off-diagonal entry for an edge is minus half the sum of the
/// cotangents of the two angles opposite it -- the identity that ties the
/// M-matrix property to the Delaunay condition, since a cotangent turns
/// negative exactly when its angle turns obtuse. Every row sums to zero,
/// because the three shape functions of a triangle sum to the constant
/// one and so their gradients sum to zero.
pub fn stiffness_matrix(mesh: &FemMesh2) -> CsrMatrix {
    let mut entries = Vec::with_capacity(9 * mesh.tris.len());
    for t in &mesh.tris {
        let (g, area) = shape_gradients(&mesh.nodes, t);
        for j in 0..3 {
            for k in 0..3 {
                entries.push((t[j], t[k], area * g[j].dot(&g[k])));
            }
        }
    }
    CsrMatrix::from_triplets(mesh.nodes.len(), mesh.nodes.len(), &entries)
}

/// The assembled consistent mass matrix.
///
/// `A/6` on the diagonal and `A/12` off it, per triangle. Its entries sum
/// to the area of the mesh, since the shape functions form a partition of
/// unity; the *lumped* alternative, which puts each row's total on its
/// diagonal, is what an explicit time integrator wants and is a different
/// matrix with the same total.
pub fn mass_matrix(mesh: &FemMesh2) -> CsrMatrix {
    let mut entries = Vec::with_capacity(9 * mesh.tris.len());
    for t in &mesh.tris {
        let area = signed_area(&mesh.nodes, t);
        for j in 0..3 {
            for k in 0..3 {
                entries.push((t[j], t[k], if j == k { area / 6.0 } else { area / 12.0 }));
            }
        }
    }
    CsrMatrix::from_triplets(mesh.nodes.len(), mesh.nodes.len(), &entries)
}

/// The gradient of a nodal field on one triangle, which is constant
/// there because the field is linear.
///
/// Returns `None` for an out-of-range triangle index or a mismatched
/// value count.
pub fn element_gradient(mesh: &FemMesh2, values: &[f64], tri: usize) -> Option<Vec2> {
    if tri >= mesh.tris.len() || values.len() != mesh.nodes.len() {
        return None;
    }
    let t = &mesh.tris[tri];
    let (g, _) = shape_gradients(&mesh.nodes, t);
    Some(Vec2::new(
        (0..3).map(|k| g[k].x * values[t[k]]).sum(),
        (0..3).map(|k| g[k].y * values[t[k]]).sum(),
    ))
}

/// The Dirichlet energy `integral |grad u|^2` of a nodal field, computed
/// exactly.
///
/// It is exact rather than quadrature-limited because the gradient is
/// constant on each triangle, so the integral is a sum of area times a
/// squared length. This is the energy norm the finite element solution
/// minimises, and the quantity that must fall when the mesh is refined.
///
/// # Errors
///
/// [`SolveError::DimensionMismatch`] if the value count does not match
/// the node count.
pub fn dirichlet_energy(mesh: &FemMesh2, values: &[f64]) -> Result<f64, SolveError> {
    if values.len() != mesh.nodes.len() {
        return Err(SolveError::DimensionMismatch {
            expected: mesh.nodes.len(),
            got: values.len(),
        });
    }
    let mut total = 0.0;
    for (i, t) in mesh.tris.iter().enumerate() {
        let g = element_gradient(mesh, values, i).expect("index and length just checked");
        total += signed_area(&mesh.nodes, t) * g.magnitude_squared();
    }
    Ok(total)
}

/// Evaluates a nodal field at an arbitrary point by locating the
/// containing triangle and interpolating barycentrically.
///
/// Returns `None` if the point lies outside every triangle, or if the
/// value count does not match the mesh. The search is linear in the
/// triangle count -- there is no spatial index here, so this is for
/// sampling an answer rather than for an inner loop.
pub fn interpolate(mesh: &FemMesh2, values: &[f64], p: Vec2) -> Option<f64> {
    if values.len() != mesh.nodes.len() {
        return None;
    }
    for t in &mesh.tris {
        let area = signed_area(&mesh.nodes, t);
        let (a, b, c) = (mesh.nodes[t[0]], mesh.nodes[t[1]], mesh.nodes[t[2]]);
        // Each barycentric coordinate is the area of the sub-triangle
        // opposite its own vertex, over the whole area. All three are
        // nonnegative exactly inside the triangle, which makes the same
        // computation serve as the containment test.
        let sub = |u: Vec2, v: Vec2| {
            0.5 * ((v.x - u.x) * (p.y - u.y) - (p.x - u.x) * (v.y - u.y)) / area
        };
        let l0 = sub(b, c);
        let l1 = sub(c, a);
        let l2 = sub(a, b);
        // The tolerance is relative to the coordinates themselves, which
        // are pure numbers of order one, so a point on a shared edge is
        // found in whichever triangle comes first rather than in
        // neither.
        if l0 >= -1e-12 && l1 >= -1e-12 && l2 >= -1e-12 {
            return Some(l0 * values[t[0]] + l1 * values[t[1]] + l2 * values[t[2]]);
        }
    }
    None
}

/// Solves the Helmholtz problem `-lap u - k^2 u = f` with Dirichlet data.
///
/// This is the same assembly as [`fem_2d_reaction_diffusion`] with a
/// negative reaction term, but it needs a different solver and the reason
/// is structural rather than numerical. Once `k^2` passes the first
/// Dirichlet eigenvalue of the domain the operator stops being positive
/// definite, and conjugate gradients -- which is a minimisation method --
/// has nothing left to minimise. A dense LU factorisation is used
/// instead, which costs `O(n^3)` in the node count and confines this
/// function to modest meshes.
///
/// At `k^2` exactly equal to an eigenvalue the operator is singular: the
/// homogeneous problem has a nonzero solution, so the inhomogeneous one
/// has either none or a whole line of them. That is resonance, not a
/// numerical accident, and it is reported as [`SolveError::Singular`].
/// Approaching an eigenvalue the response grows like the reciprocal of
/// the distance to it.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for non-finite data,
/// [`SolveError::Singular`] at or extremely close to a resonance.
pub fn fem_2d_helmholtz(
    mesh: &FemMesh2,
    k: f64,
    f: &dyn Fn(Vec2) -> f64,
    dirichlet: &dyn Fn(Vec2) -> Option<f64>,
) -> Result<Vec<f64>, SolveError> {
    if !k.is_finite() {
        return Err(SolveError::InvalidArgument("the wavenumber must be finite"));
    }
    let n = mesh.nodes.len();
    let asm = assemble(mesh, &|_| -k * k, f)?;
    let mut fixed = vec![None; n];
    for &b in &mesh.boundary {
        if let Some(g) = dirichlet(mesh.nodes[b]) {
            if !g.is_finite() {
                return Err(SolveError::InvalidArgument("boundary data must be finite"));
            }
            fixed[b] = Some(g);
        }
    }
    let (entries, rhs) = apply_dirichlet(n, &asm.entries, &asm.load, &fixed);
    let mut dense = Matrix::zeros(n, n);
    for (i, j, v) in entries {
        dense.set(i, j, dense.get(i, j) + v);
    }
    crate::linalg::lu::solve(&dense, &rhs)
}

/// The `count` smallest eigenvalues of the Dirichlet Laplacian on the
/// mesh -- the squared frequencies of a drum clamped at its rim.
///
/// The discrete problem is the generalised one `K phi = lambda M phi`
/// over the interior nodes, solved by transforming it to a standard
/// symmetric problem through the Cholesky factor of the mass matrix.
/// Using the consistent mass matrix rather than a lumped one matters
/// here: lumping shifts the eigenvalues downwards, and it is precisely
/// their being *upper* bounds that makes them useful.
///
/// That bound is the property worth knowing. The discrete eigenvalues
/// come from the Rayleigh quotient minimised over a subspace of the true
/// admissible space, and a minimum over less is never smaller, so every
/// computed eigenvalue is an upper bound on the true one and refining the
/// mesh can only lower it. A method whose eigenvalues approach the answer
/// from below has a defect, however good its error looks.
///
/// The dense eigensolver is `O(n^3)` in the interior node count, so this
/// is for meshes of hundreds of nodes rather than thousands.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] if `count` is zero or exceeds the
/// number of interior nodes; [`SolveError::NotPositiveDefinite`] if the
/// mass matrix fails to factor, and whatever the eigensolver reports.
pub fn fem_eigenvalues_drum(mesh: &FemMesh2, count: usize) -> Result<Vec<f64>, SolveError> {
    Ok(fem_eigenmodes_drum(mesh, count)?.0)
}

/// The `count` lowest drum modes: eigenvalues and the matching nodal
/// eigenvectors, the latter given over all nodes with zeros on the
/// clamped boundary.
///
/// Eigenvectors are normalised so that the mass-weighted norm
/// `phi^T M phi` is one, which is the discrete form of normalising the
/// mode shape in `L2`.
///
/// # Errors
///
/// As [`fem_eigenvalues_drum`].
pub fn fem_eigenmodes_drum(
    mesh: &FemMesh2,
    count: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), SolveError> {
    let n = mesh.nodes.len();
    let on_boundary: std::collections::HashSet<usize> = mesh.boundary.iter().copied().collect();
    let interior: Vec<usize> = (0..n).filter(|i| !on_boundary.contains(i)).collect();
    let m = interior.len();
    if count == 0 {
        return Err(SolveError::InvalidArgument("need at least one mode"));
    }
    if count > m {
        return Err(SolveError::InvalidArgument("the mesh has fewer interior nodes than modes"));
    }
    let mut index = vec![usize::MAX; n];
    for (slot, &i) in interior.iter().enumerate() {
        index[i] = slot;
    }
    let stiff = stiffness_matrix(mesh);
    let mass = mass_matrix(mesh);
    let restrict = |src: &CsrMatrix| {
        let mut d = Matrix::zeros(m, m);
        for i in 0..n {
            if index[i] == usize::MAX {
                continue;
            }
            for idx in src.row_ptr[i]..src.row_ptr[i + 1] {
                let j = src.col_idx[idx];
                if index[j] != usize::MAX {
                    let (r, c) = (index[i], index[j]);
                    d.set(r, c, d.get(r, c) + src.vals[idx]);
                }
            }
        }
        d
    };
    let k_dense = restrict(&stiff);
    let m_dense = restrict(&mass);
    // M = L L^T, and the substitution phi = L^-T y turns
    // K phi = lambda M phi into (L^-1 K L^-T) y = lambda y.
    let l = crate::linalg::cholesky::cholesky(&m_dense)?;
    let y = forward_substitute_columns(&l, &k_dense)?;
    // C^T = L^-1 (L^-1 K)^T, and C is symmetric, so symmetrising here
    // removes the rounding asymmetry the two solves introduce rather
    // than leaving the eigensolver to cope with it.
    let z = forward_substitute_columns(&l, &y.transpose())?;
    let c = Matrix::from_fn(m, m, |i, j| 0.5 * (z.get(j, i) + z.get(i, j)));
    let eig = crate::linalg::eigen::eigen_symmetric(&c, 1e-12, 200)?;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eig.values[a].total_cmp(&eig.values[b]));
    let mut values = Vec::with_capacity(count);
    let mut modes = Vec::with_capacity(count);
    for &slot in order.iter().take(count) {
        values.push(eig.values[slot]);
        let y: Vec<f64> = (0..m).map(|r| eig.vectors.get(r, slot)).collect();
        // phi = L^-T y, a back substitution against the transpose.
        let mut phi = y;
        for r in (0..m).rev() {
            let mut acc = phi[r];
            for c2 in (r + 1)..m {
                acc -= l.get(c2, r) * phi[c2];
            }
            phi[r] = acc / l.get(r, r);
        }
        let mut full = vec![0.0; n];
        for (slot, &i) in interior.iter().enumerate() {
            full[i] = phi[slot];
        }
        // Normalise in the mass norm, and fix the sign so that a mode is
        // reproducible: an eigenvector is only defined up to a scale.
        let norm = mass_quadratic_form(&mass, &full).max(0.0).sqrt();
        if norm > 0.0 {
            let biggest = full
                .iter()
                .copied()
                .fold(0.0f64, |a, v| if v.abs() > a.abs() { v } else { a });
            let sign = if biggest < 0.0 { -1.0 } else { 1.0 };
            for v in &mut full {
                *v *= sign / norm;
            }
        }
        modes.push(full);
    }
    Ok((values, modes))
}

/// `v^T M v` for a sparse `M`.
fn mass_quadratic_form(mass: &CsrMatrix, v: &[f64]) -> f64 {
    let mv = mass.mul_vec(v);
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum()
}

/// Solves `L X = B` for `X`, column by column, with `L` lower
/// triangular.
fn forward_substitute_columns(l: &Matrix, b: &Matrix) -> Result<Matrix, SolveError> {
    let n = l.rows;
    if b.rows != n {
        return Err(SolveError::DimensionMismatch { expected: n, got: b.rows });
    }
    let mut x = Matrix::zeros(n, b.cols);
    for col in 0..b.cols {
        for r in 0..n {
            let mut acc = b.get(r, col);
            for c in 0..r {
                acc -= l.get(r, c) * x.get(c, col);
            }
            let d = l.get(r, r);
            if d == 0.0 {
                return Err(SolveError::Singular);
            }
            x.set(r, col, acc / d);
        }
    }
    Ok(x)
}

/// The plane-stress constitutive matrix, relating
/// `(sigma_x, sigma_y, tau)` to `(eps_x, eps_y, gamma)`.
fn plane_stress_d(e: f64, nu: f64) -> [[f64; 3]; 3] {
    let k = e / (1.0 - nu * nu);
    [[k, k * nu, 0.0], [k * nu, k, 0.0], [0.0, 0.0, k * 0.5 * (1.0 - nu)]]
}

/// The strain-displacement matrix of a constant-strain triangle, three
/// strains by six degrees of freedom.
fn strain_matrix(g: &[Vec2; 3]) -> [[f64; 6]; 3] {
    let mut b = [[0.0; 6]; 3];
    for k in 0..3 {
        b[0][2 * k] = g[k].x;
        b[1][2 * k + 1] = g[k].y;
        b[2][2 * k] = g[k].y;
        b[2][2 * k + 1] = g[k].x;
    }
    b
}

/// Solves the plane-stress elasticity problem on the mesh.
///
/// `loads` are point forces applied at nodes and `fixed` prescribes
/// displacements at nodes, both components at once. Unit thickness is
/// assumed throughout, so a force is a force per unit thickness.
///
/// # The constant-strain triangle
///
/// Displacement is linear on each triangle, so strain -- its gradient --
/// is constant there, and so is stress. That makes the element matrix
/// `A B^T D B` with no quadrature, exactly as for the Laplacian, and it
/// makes the stress field piecewise constant and discontinuous across
/// every edge. The discontinuity is not a bug to be smoothed away
/// silently: its size is an error estimate, and averaging it to the
/// nodes before showing it to anyone is how a coarse mesh comes to look
/// convincing.
///
/// # What has to be pinned
///
/// The stiffness matrix has a three-dimensional kernel: two translations
/// and one infinitesimal rotation. Prescribing fewer than three
/// independent degrees of freedom leaves the body free to move without
/// straining, and the system is singular no matter how many loads are
/// applied. This is checked directly.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a non-positive modulus, a
/// Poisson's ratio outside `(-1, 0.5)`, an out-of-range node index, or
/// non-finite data; [`SolveError::Singular`] if the constraints leave a
/// rigid body motion free.
pub fn fem_2d_elasticity_plane_stress(
    mesh: &FemMesh2,
    e: f64,
    nu: f64,
    loads: &[(usize, Vec2)],
    fixed: &[(usize, Vec2)],
) -> Result<Vec<Vec2>, SolveError> {
    let n = mesh.nodes.len();
    if !e.is_finite() || e <= 0.0 {
        return Err(SolveError::InvalidArgument("Young's modulus must be positive"));
    }
    if !nu.is_finite() || nu <= -1.0 || nu >= 0.5 {
        return Err(SolveError::InvalidArgument("Poisson's ratio must lie in (-1, 1/2)"));
    }
    let d = plane_stress_d(e, nu);
    let mut entries = Vec::with_capacity(36 * mesh.tris.len());
    for t in &mesh.tris {
        let (g, area) = shape_gradients(&mesh.nodes, t);
        let b = strain_matrix(&g);
        // db = D B, then the element matrix is A B^T (D B).
        let mut db = [[0.0f64; 6]; 3];
        for r in 0..3 {
            for c in 0..6 {
                db[r][c] = (0..3).map(|k| d[r][k] * b[k][c]).sum();
            }
        }
        for i in 0..6 {
            for j in 0..6 {
                let v: f64 = (0..3).map(|k| b[k][i] * db[k][j]).sum();
                entries.push((2 * t[i / 2] + i % 2, 2 * t[j / 2] + j % 2, area * v));
            }
        }
    }
    let mut rhs = vec![0.0; 2 * n];
    for &(node, force) in loads {
        if node >= n {
            return Err(SolveError::InvalidArgument("load applied to a node that is not there"));
        }
        if !(force.x.is_finite() && force.y.is_finite()) {
            return Err(SolveError::InvalidArgument("loads must be finite"));
        }
        rhs[2 * node] += force.x;
        rhs[2 * node + 1] += force.y;
    }
    let mut pinned = vec![None; 2 * n];
    for &(node, u) in fixed {
        if node >= n {
            return Err(SolveError::InvalidArgument("constraint on a node that is not there"));
        }
        if !(u.x.is_finite() && u.y.is_finite()) {
            return Err(SolveError::InvalidArgument("prescribed displacements must be finite"));
        }
        pinned[2 * node] = Some(u.x);
        pinned[2 * node + 1] = Some(u.y);
    }
    // Two translations and an infinitesimal rotation about the origin.
    // If all three survive the constraints the body can move without
    // straining and the system is singular whatever the loads are.
    let free_motion = |field: &dyn Fn(Vec2) -> Vec2| {
        (0..n).all(|i| {
            let v = field(mesh.nodes[i]);
            (pinned[2 * i].is_none() || v.x == 0.0) && (pinned[2 * i + 1].is_none() || v.y == 0.0)
        })
    };
    if free_motion(&|_| Vec2::new(1.0, 0.0))
        || free_motion(&|_| Vec2::new(0.0, 1.0))
        || free_motion(&|p| Vec2::new(-p.y, p.x))
    {
        return Err(SolveError::Singular);
    }
    let (entries, rhs) = apply_dirichlet(2 * n, &entries, &rhs, &pinned);
    let matrix = CsrMatrix::from_triplets(2 * n, 2 * n, &entries);
    let u = pcg_jacobi(&matrix, &rhs, 1e-14, 200 * n + 2000)?;
    Ok((0..n).map(|i| Vec2::new(u[2 * i], u[2 * i + 1])).collect())
}

/// The constant strain `(eps_x, eps_y, gamma)` of one triangle, given a
/// nodal displacement field.
///
/// `gamma` is the engineering shear strain, twice the tensor component.
/// Returns `None` for an out-of-range index or a mismatched field.
pub fn element_strain(mesh: &FemMesh2, u: &[Vec2], tri: usize) -> Option<[f64; 3]> {
    if tri >= mesh.tris.len() || u.len() != mesh.nodes.len() {
        return None;
    }
    let t = &mesh.tris[tri];
    let (g, _) = shape_gradients(&mesh.nodes, t);
    let b = strain_matrix(&g);
    let local = [u[t[0]].x, u[t[0]].y, u[t[1]].x, u[t[1]].y, u[t[2]].x, u[t[2]].y];
    let mut out = [0.0; 3];
    for (r, slot) in out.iter_mut().enumerate() {
        *slot = (0..6).map(|c| b[r][c] * local[c]).sum();
    }
    Some(out)
}

/// The constant stress `(sigma_x, sigma_y, tau)` of one triangle.
///
/// Returns `None` for an out-of-range index, a mismatched field, or
/// material constants outside their admissible ranges.
pub fn element_stress(
    mesh: &FemMesh2,
    u: &[Vec2],
    e: f64,
    nu: f64,
    tri: usize,
) -> Option<[f64; 3]> {
    if !e.is_finite() || e <= 0.0 || !nu.is_finite() || nu <= -1.0 || nu >= 0.5 {
        return None;
    }
    let strain = element_strain(mesh, u, tri)?;
    let d = plane_stress_d(e, nu);
    let mut out = [0.0; 3];
    for (r, slot) in out.iter_mut().enumerate() {
        *slot = (0..3).map(|k| d[r][k] * strain[k]).sum();
    }
    Some(out)
}

/// The total strain energy `(1/2) integral sigma : eps`.
///
/// At equilibrium this is half the work the applied loads do, which is
/// Clapeyron's theorem and follows from nothing more than the stiffness
/// matrix being symmetric.
///
/// # Errors
///
/// [`SolveError::DimensionMismatch`] for a mismatched field and
/// [`SolveError::InvalidArgument`] for invalid material constants.
pub fn strain_energy(
    mesh: &FemMesh2,
    u: &[Vec2],
    e: f64,
    nu: f64,
) -> Result<f64, SolveError> {
    if u.len() != mesh.nodes.len() {
        return Err(SolveError::DimensionMismatch { expected: mesh.nodes.len(), got: u.len() });
    }
    let mut total = 0.0;
    for (i, t) in mesh.tris.iter().enumerate() {
        let strain = element_strain(mesh, u, i).expect("index and length just checked");
        let stress = element_stress(mesh, u, e, nu, i)
            .ok_or(SolveError::InvalidArgument("invalid material constants"))?;
        let density: f64 = (0..3).map(|k| stress[k] * strain[k]).sum();
        total += 0.5 * signed_area(&mesh.nodes, t) * density;
    }
    Ok(total)
}

/// The von Mises equivalent stress of each triangle, given a nodal
/// displacement field.
///
/// One value per triangle, not per node: the strain of a linear
/// displacement field is constant on an element and discontinuous across
/// its edges. That discontinuity is not a bug to be smoothed away
/// silently -- its size is an error estimate, and averaging it to the
/// nodes before showing it to anyone is how a coarse mesh comes to look
/// convincing.
///
/// In plane stress the out-of-plane stress is zero rather than free, so
/// the equivalent stress is
/// `sqrt(sx^2 - sx sy + sy^2 + 3 tau^2)`. A consequence worth noticing:
/// equal biaxial tension `sx = sy = s` gives `|s|`, not zero. The
/// three-dimensional intuition that hydrostatic stress cannot yield a
/// material does not survive into plane stress, because a state that is
/// hydrostatic *in plane* has a free surface out of it and so is not
/// hydrostatic at all.
///
/// # Errors
///
/// [`SolveError::DimensionMismatch`] if the displacement count does not
/// match the node count, and [`SolveError::InvalidArgument`] for invalid
/// material constants.
pub fn von_mises_stress(
    mesh: &FemMesh2,
    u: &[Vec2],
    e: f64,
    nu: f64,
) -> Result<Vec<f64>, SolveError> {
    if u.len() != mesh.nodes.len() {
        return Err(SolveError::DimensionMismatch { expected: mesh.nodes.len(), got: u.len() });
    }
    (0..mesh.tris.len())
        .map(|i| {
            let s = element_stress(mesh, u, e, nu, i)
                .ok_or(SolveError::InvalidArgument("invalid material constants"))?;
            Ok((s[0] * s[0] - s[0] * s[1] + s[1] * s[1] + 3.0 * s[2] * s[2]).max(0.0).sqrt())
        })
        .collect()
}

/// Marches the heat equation `u_t = alpha lap u + f` with the
/// `theta` scheme, returning `steps + 1` snapshots starting from the
/// initial field.
///
/// The step solves
/// `(M + theta alpha dt K) u_next = (M - (1-theta) alpha dt K) u + dt F`.
/// `theta = 0` is forward Euler, `1` backward Euler, `1/2`
/// Crank-Nicolson.
///
/// # Stability, and the difference between A-stable and L-stable
///
/// Applied to a discrete eigenmode the scheme multiplies its amplitude
/// by `(1 - (1-theta) a) / (1 + theta a)` each step, with
/// `a = alpha lambda dt`. For `theta >= 1/2` that factor has magnitude
/// below one for every positive `a`, which is A-stability, and forward
/// Euler instead needs `a < 2`.
///
/// Crank-Nicolson is A-stable but *not* L-stable: as `a` grows its factor
/// tends to `-1`, not to zero. A mode too stiff to resolve therefore
/// survives while flipping sign every step, which is why a discontinuous
/// initial condition rings under Crank-Nicolson and why the usual remedy
/// is to take the first couple of steps with backward Euler, whose
/// factor does tend to zero. That contrast is asserted in the tests.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a mismatched initial field, a
/// non-positive step, a `theta` outside `[0, 1]`, a negative diffusivity
/// or non-finite data; whatever the linear solver reports otherwise.
#[allow(clippy::too_many_arguments)]
pub fn fem_2d_heat_transient(
    mesh: &FemMesh2,
    initial: &[f64],
    alpha: f64,
    dt: f64,
    steps: usize,
    theta: f64,
    source: &dyn Fn(Vec2) -> f64,
    dirichlet: &dyn Fn(Vec2) -> Option<f64>,
) -> Result<Vec<Vec<f64>>, SolveError> {
    let n = mesh.nodes.len();
    if initial.len() != n {
        return Err(SolveError::DimensionMismatch { expected: n, got: initial.len() });
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SolveError::InvalidArgument("the time step must be positive"));
    }
    if !theta.is_finite() || !(0.0..=1.0).contains(&theta) {
        return Err(SolveError::InvalidArgument("theta must lie in [0, 1]"));
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(SolveError::InvalidArgument("the diffusivity must be nonnegative"));
    }
    if initial.iter().any(|v| !v.is_finite()) {
        return Err(SolveError::InvalidArgument("the initial field must be finite"));
    }
    let asm = assemble(mesh, &|_| 0.0, source)?;
    let stiff = stiffness_matrix(mesh);
    let mass = mass_matrix(mesh);
    let triplets = |m: &CsrMatrix, k: f64| -> Vec<(usize, usize, f64)> {
        (0..n)
            .flat_map(|i| {
                (m.row_ptr[i]..m.row_ptr[i + 1]).map(move |idx| (i, idx))
            })
            .map(|(i, idx)| (i, m.col_idx[idx], k * m.vals[idx]))
            .collect()
    };
    let mut fixed = vec![None; n];
    for &b in &mesh.boundary {
        if let Some(g) = dirichlet(mesh.nodes[b]) {
            if !g.is_finite() {
                return Err(SolveError::InvalidArgument("boundary data must be finite"));
            }
            fixed[b] = Some(g);
        }
    }
    // The implicit side is fixed for the whole march, so it is
    // assembled and factored into a CSR matrix once.
    let mut lhs = triplets(&mass, 1.0);
    lhs.extend(triplets(&stiff, theta * alpha * dt));
    let explicit_mass = triplets(&mass, 1.0);
    let explicit_stiff = triplets(&stiff, -(1.0 - theta) * alpha * dt);
    let mut current = initial.to_vec();
    // Snap the boundary onto its prescribed value before the first step
    // so that the recorded history is consistent from the outset rather
    // than at step one.
    for (i, slot) in fixed.iter().enumerate() {
        if let Some(g) = *slot {
            current[i] = g;
        }
    }
    let mut history = vec![current.clone()];
    let apply = |entries: &[(usize, usize, f64)], v: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n];
        for &(i, j, k) in entries {
            out[i] += k * v[j];
        }
        out
    };
    for _ in 0..steps {
        let a = apply(&explicit_mass, &current);
        let b = apply(&explicit_stiff, &current);
        let rhs: Vec<f64> = (0..n).map(|i| a[i] + b[i] + dt * asm.load[i]).collect();
        let (entries, rhs) = apply_dirichlet(n, &lhs, &rhs, &fixed);
        let matrix = CsrMatrix::from_triplets(n, n, &entries);
        current = pcg_jacobi(&matrix, &rhs, 1e-14, 20 * n + 500)?;
        history.push(current.clone());
    }
    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    /// V - E + T = 1 for any triangulated simply connected region: the
    /// Euler characteristic of a disk, with the outer face excluded.
    fn euler(mesh: &FemMesh2) -> i64 {
        mesh.nodes.len() as i64 - mesh.edge_count() as i64 + mesh.tris.len() as i64
    }

    #[test]
    fn the_rectangle_mesh_is_a_conforming_triangulation() {
        let m = FemMesh2::rect(2.0, 3.0, 4, 5).unwrap();
        assert_eq!(m.nodes.len(), 5 * 6);
        assert_eq!(m.tris.len(), 2 * 4 * 5);
        assert!((m.area() - 6.0).abs() < 1e-12);
        assert_eq!(euler(&m), 1);
        // The boundary of a 4x5 grid is its perimeter of nodes.
        assert_eq!(m.boundary.len(), 2 * (4 + 5));
        // The cells are 0.5 by 0.6, split along a diagonal, so every
        // triangle is right-angled with those legs and the smallest
        // angle is atan of their ratio -- 45 degrees only when the
        // cells are square, which these are not.
        let (dx, dy): (f64, f64) = (2.0 / 4.0, 3.0 / 5.0);
        assert!((m.quality_min_angle() - (dx / dy).atan()).abs() < 1e-12);
        let square = FemMesh2::rect(1.0, 1.0, 3, 3).unwrap();
        assert!((square.quality_min_angle() - PI / 4.0).abs() < 1e-12);
        for t in &m.tris {
            assert!(signed_area(&m.nodes, t) > 0.0, "a triangle came out clockwise");
        }
    }

    #[test]
    fn the_disk_mesh_closes_up_and_approaches_the_right_area() {
        for n in [1usize, 2, 5, 9] {
            let m = FemMesh2::disk(2.0, n).unwrap();
            assert_eq!(m.nodes.len(), 1 + 3 * n * (n + 1));
            assert_eq!(euler(&m), 1, "{n} rings");
            // The outer ring is the boundary and nothing else is.
            assert_eq!(m.boundary.len(), 6 * n);
            // A polygon inscribed in the disk, so the area is below the
            // circle's and rises towards it.
            let exact = PI * 4.0;
            assert!(m.area() < exact, "{n} rings overshot the disk");
            assert!(m.area() > exact * (1.0 - 4.0 / (n * n) as f64 - 0.3));
        }
        let coarse = FemMesh2::disk(1.0, 3).unwrap();
        let fine = FemMesh2::disk(1.0, 12).unwrap();
        assert!(fine.area() > coarse.area(), "refining the disk lost area");
    }

    #[test]
    fn uniform_refinement_multiplies_the_triangles_and_keeps_the_shape() {
        let m = FemMesh2::rect(1.0, 1.0, 3, 2).unwrap();
        let (v, e, t) = (m.nodes.len(), m.edge_count(), m.tris.len());
        let r = m.refine_uniform();
        // One new node per edge, four children per triangle.
        assert_eq!(r.nodes.len(), v + e);
        assert_eq!(r.tris.len(), 4 * t);
        assert!((r.area() - m.area()).abs() < 1e-12);
        assert_eq!(euler(&r), 1);
        // The four children are similar to the parent, so the worst
        // angle in the mesh is exactly what it was.
        assert!((r.quality_min_angle() - m.quality_min_angle()).abs() < 1e-14);
    }

    #[test]
    fn a_degenerate_or_inconsistent_mesh_is_refused() {
        let square =
            vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)];
        assert_eq!(FemMesh2::new(square.clone(), vec![]), Err(GeomError::Empty));
        assert!(FemMesh2::new(square.clone(), vec![[0, 1, 5]]).is_err());
        assert!(FemMesh2::new(square.clone(), vec![[0, 1, 1]]).is_err());
        let flat = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
        assert_eq!(
            FemMesh2::new(flat, vec![[0, 1, 2]]),
            Err(GeomError::Degenerate("zero-area triangle"))
        );
        // Three triangles on one edge is not a surface.
        let fan = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        assert_eq!(
            FemMesh2::new(fan, vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]]),
            Err(GeomError::NotManifold)
        );
        assert!(FemMesh2::rect(0.0, 1.0, 2, 2).is_err());
        assert!(FemMesh2::rect(1.0, 1.0, 0, 2).is_err());
        assert!(FemMesh2::disk(-1.0, 2).is_err());
        assert!(FemMesh2::disk(1.0, 0).is_err());
        assert!(FemMesh2::from_delaunay(&[Vec2::ZERO, Vec2::new(1.0, 0.0)]).is_err());
    }

    #[test]
    fn a_clockwise_triangle_is_reoriented_rather_than_rejected() {
        let p = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        let m = FemMesh2::new(p, vec![[0, 2, 1]]).unwrap();
        assert!(signed_area(&m.nodes, &m.tris[0]) > 0.0);
        assert!((m.area() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn a_linear_solution_is_reproduced_exactly() {
        // The two-dimensional patch test. A linear field is in the
        // element space, its Laplacian is zero, and the method must
        // return it untouched from its boundary values alone.
        let m = FemMesh2::rect(2.0, 1.0, 6, 4).unwrap();
        let exact = |p: Vec2| 1.0 + 2.0 * p.x - 3.0 * p.y;
        let u = fem_2d_poisson(&m, &|_| 0.0, &|p| Some(exact(p))).unwrap();
        for (i, &got) in u.iter().enumerate() {
            let want = exact(m.nodes[i]);
            assert!((got - want).abs() < 1e-10, "node {i} was off by {}", got - want);
        }
    }

    #[test]
    fn poisson_converges_at_second_order_on_the_square() {
        let u = |p: Vec2| (PI * p.x).sin() * (PI * p.y).sin();
        let f = |p: Vec2| 2.0 * PI * PI * (PI * p.x).sin() * (PI * p.y).sin();
        let mut previous = f64::INFINITY;
        let mut ratios = Vec::new();
        for n in [4usize, 8, 16, 32] {
            let m = FemMesh2::rect(1.0, 1.0, n, n).unwrap();
            let v = fem_2d_poisson(&m, &f, &|_| Some(0.0)).unwrap();
            let err = v
                .iter()
                .enumerate()
                .map(|(i, &g)| (g - u(m.nodes[i])).abs())
                .fold(0.0, f64::max);
            if previous.is_finite() {
                ratios.push(previous / err);
            }
            previous = err;
        }
        for r in &ratios {
            assert!((r - 4.0).abs() < 0.4, "halving h cut the error by {r}, not 4");
        }
    }

    #[test]
    fn a_pure_neumann_problem_is_singular_and_a_reaction_term_fixes_it() {
        let m = FemMesh2::rect(1.0, 1.0, 4, 4).unwrap();
        assert_eq!(fem_2d_poisson(&m, &|_| 1.0, &|_| None), Err(SolveError::Singular));
        let v = fem_2d_reaction_diffusion(&m, &|_| 2.0, &|_| 2.0, &|_| None).unwrap();
        // -lap u + 2u = 2 with no flux anywhere has the constant
        // solution u = 1, and the constant is in the element space.
        for &g in &v {
            assert!((g - 1.0).abs() < 1e-9, "got {g}");
        }
    }

    #[test]
    fn interpolation_reproduces_the_nodal_values_and_refuses_the_outside() {
        let m = FemMesh2::rect(1.0, 1.0, 2, 2).unwrap();
        let values: Vec<f64> = m.nodes.iter().map(|p| 3.0 * p.x - p.y).collect();
        for (i, &p) in m.nodes.iter().enumerate() {
            let got = interpolate(&m, &values, p).unwrap();
            assert!((got - values[i]).abs() < 1e-12);
        }
        // A linear field interpolates exactly anywhere inside.
        let p = Vec2::new(0.37, 0.81);
        assert!((interpolate(&m, &values, p).unwrap() - (3.0 * 0.37 - 0.81)).abs() < 1e-12);
        assert!(interpolate(&m, &values, Vec2::new(2.0, 2.0)).is_none());
        assert!(interpolate(&m, &values[..3], p).is_none());
    }

    #[test]
    fn the_square_drum_hears_its_analytic_frequencies_from_above() {
        // On the unit square the Dirichlet eigenvalues are
        // pi^2 (m^2 + n^2). The discrete ones are Rayleigh quotients
        // minimised over a subspace, so each is an upper bound, and
        // refining lowers it.
        let exact = [
            2.0 * PI * PI,
            5.0 * PI * PI,
            5.0 * PI * PI,
            8.0 * PI * PI,
            10.0 * PI * PI,
        ];
        let mut previous: Option<Vec<f64>> = None;
        let mut errors = Vec::new();
        for n in [6usize, 12] {
            let m = FemMesh2::rect(1.0, 1.0, n, n).unwrap();
            let got = fem_eigenvalues_drum(&m, 5).unwrap();
            for (i, (&g, &e)) in got.iter().zip(exact.iter()).enumerate() {
                assert!(g >= e - 1e-8, "mode {i} came in below the exact value: {g} < {e}");
                assert!((g - e) / e < 0.6, "mode {i} was {g}, wanted {e}");
            }
            if let Some(coarse) = &previous {
                for (i, (&fine, &c)) in got.iter().zip(coarse.iter()).enumerate() {
                    assert!(fine <= c + 1e-9, "refining raised mode {i}");
                }
            }
            errors.push(got[0] - exact[0]);
            previous = Some(got);
        }
        // Second order, which is the statement that identifies the
        // space -- an absolute tolerance on one mesh would not.
        let ratio = errors[0] / errors[1];
        assert!((ratio - 4.0).abs() < 0.5, "halving h cut the eigenvalue error by {ratio}");
        // The doubled eigenvalue 5 pi^2 splits, because the mesh cuts
        // every cell along the same diagonal and so is not symmetric
        // under exchanging x and y. The split is a mesh artefact of the
        // same order as the error itself, not a physical degeneracy
        // lifting.
        let fine = previous.unwrap();
        let split = (fine[2] - fine[1]) / exact[1];
        assert!(split > 0.0, "the pair did not split at all");
        assert!(split < 0.05, "the pair split by {split}, far more than the discretisation error");
    }

    #[test]
    fn the_circular_drum_hears_the_zeros_of_the_bessel_functions() {
        // The modes of a clamped circular membrane are J_m(j_{m,k} r/R),
        // so the eigenvalues are (j_{m,k}/R)^2. Checking against the
        // crate's own Bessel zeros ties the finite element solver to the
        // analytic membrane rather than to a hard-coded table.
        let r = 1.0;
        let m = FemMesh2::disk(r, 8).unwrap();
        let got = fem_eigenvalues_drum(&m, 5).unwrap();
        let j0 = crate::special::bessel::bessel_j_zeros(0, 2);
        let j1 = crate::special::bessel::bessel_j_zeros(1, 1);
        let j2 = crate::special::bessel::bessel_j_zeros(2, 1);
        // The order is not by Bessel index. j_{2,1} = 5.136 comes in
        // below j_{0,2} = 5.520, so the fourth and fifth modes of a
        // circular drum are the doubly degenerate two-nodal-diameter
        // pair, and the second radially symmetric mode is only sixth.
        assert!(j2[0] < j0[1], "the Bessel zeros are not ordered as assumed");
        let exact = [j0[0], j1[0], j1[0], j2[0], j2[0]].map(|z| (z / r).powi(2));
        for (i, (&g, &e)) in got.iter().zip(exact.iter()).enumerate() {
            assert!(g >= e - 1e-6, "mode {i} came in below the exact value");
            assert!((g - e) / e < 0.08, "mode {i} was {g}, wanted {e}");
        }
        // Each degenerate pair is one mode shape turned through a right
        // angle, so the two must agree far more closely than either
        // agrees with the analytic value.
        for (a, b) in [(1usize, 2usize), (3, 4)] {
            let split = (got[b] - got[a]).abs() / got[a];
            assert!(split < 0.01, "the pair {a},{b} split by {split}");
        }
    }

    #[test]
    fn helmholtz_reduces_to_poisson_at_zero_wavenumber() {
        let m = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let f = |p: Vec2| 1.0 + p.x;
        let g = |p: Vec2| Some(p.y * p.y);
        let a = fem_2d_poisson(&m, &f, &g).unwrap();
        let b = fem_2d_helmholtz(&m, 0.0, &f, &g).unwrap();
        for i in 0..m.nodes.len() {
            assert!((a[i] - b[i]).abs() < 1e-9 * (1.0 + a[i].abs()), "node {i}");
        }
    }

    #[test]
    fn the_helmholtz_response_diverges_and_flips_sign_across_a_resonance() {
        // Expanded in the drum modes, the response to a source is a sum
        // of terms with 1/(lambda_j - k^2). Approaching the fundamental
        // from below the first term dominates and is positive; from
        // above it dominates and is negative. Both the divergence and
        // the sign change are properties of the operator, not of the
        // discretisation.
        let m = FemMesh2::rect(1.0, 1.0, 8, 8).unwrap();
        let lambda = fem_eigenvalues_drum(&m, 1).unwrap()[0];
        let middle = m.nodes.iter().position(|p| {
            (p.x - 0.5).abs() < 1e-12 && (p.y - 0.5).abs() < 1e-12
        }).unwrap();
        let mut previous = 0.0;
        for gap in [0.2f64, 0.05, 0.01] {
            let k = (lambda * (1.0 - gap)).sqrt();
            let u = fem_2d_helmholtz(&m, k, &|_| 1.0, &|_| Some(0.0)).unwrap();
            assert!(u[middle] > previous, "the response did not grow approaching resonance");
            previous = u[middle];
        }
        let above = fem_2d_helmholtz(
            &m,
            (lambda * 1.01).sqrt(),
            &|_| 1.0,
            &|_| Some(0.0),
        )
        .unwrap();
        assert!(above[middle] < 0.0, "the response did not flip sign past the resonance");
        assert!(above[middle].abs() > 1.0, "the response past resonance was not large");
    }

    #[test]
    fn eigenmodes_are_mass_normalised_and_orthogonal() {
        let m = FemMesh2::rect(1.0, 1.0, 7, 6).unwrap();
        let (values, modes) = fem_eigenmodes_drum(&m, 4).unwrap();
        let mass = mass_matrix(&m);
        let stiff = stiffness_matrix(&m);
        for (i, mode) in modes.iter().enumerate() {
            assert!((mass_quadratic_form(&mass, mode) - 1.0).abs() < 1e-9);
            // The Rayleigh quotient of a mode is its own eigenvalue.
            let kv = stiff.mul_vec(mode);
            let rayleigh: f64 = mode.iter().zip(kv.iter()).map(|(a, b)| a * b).sum();
            assert!((rayleigh - values[i]).abs() < 1e-7 * values[i], "mode {i}");
            // Clamped at the rim.
            for &b in &m.boundary {
                assert_eq!(mode[b], 0.0);
            }
        }
        // Distinct modes are M-orthogonal.
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                if (values[i] - values[j]).abs() < 1e-6 * values[j] {
                    continue;
                }
                let mv = mass.mul_vec(&modes[j]);
                let dot: f64 = modes[i].iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-8, "modes {i} and {j} overlapped by {dot}");
            }
        }
    }

    #[test]
    fn the_eigenvalue_helpers_refuse_impossible_requests() {
        let m = FemMesh2::rect(1.0, 1.0, 2, 2).unwrap();
        assert!(fem_eigenvalues_drum(&m, 0).is_err());
        // A 2x2 grid has exactly one interior node.
        assert!(fem_eigenvalues_drum(&m, 1).is_ok());
        assert!(fem_eigenvalues_drum(&m, 2).is_err());
        assert!(fem_2d_helmholtz(&m, f64::NAN, &|_| 1.0, &|_| Some(0.0)).is_err());
        assert!(fem_2d_helmholtz(&m, 1.0, &|_| 1.0, &|_| Some(f64::NAN)).is_err());
    }

    /// Boundary nodes pinned to a displacement field.
    fn pin_boundary(m: &FemMesh2, field: &dyn Fn(Vec2) -> Vec2) -> Vec<(usize, Vec2)> {
        m.boundary.iter().map(|&b| (b, field(m.nodes[b]))).collect()
    }

    #[test]
    fn rigid_body_motions_produce_no_stress() {
        // Two translations and an infinitesimal rotation span the kernel
        // of the stiffness matrix. A method that strained under a
        // rotation would be wrong in a way no convergence study catches,
        // because the error would be first order in the rotation and
        // would look like a genuine load.
        let m = FemMesh2::rect(2.0, 1.0, 4, 3).unwrap();
        for field in [
            &(|_: Vec2| Vec2::new(0.7, 0.0)) as &dyn Fn(Vec2) -> Vec2,
            &|_: Vec2| Vec2::new(0.0, -1.3),
            &|p: Vec2| Vec2::new(-0.4 * p.y, 0.4 * p.x),
            &|p: Vec2| Vec2::new(2.0 - 0.4 * p.y, 0.4 * p.x - 1.0),
        ] {
            let u: Vec<Vec2> = m.nodes.iter().map(|&p| field(p)).collect();
            for (i, s) in von_mises_stress(&m, &u, 210e9, 0.3).unwrap().iter().enumerate() {
                assert!(*s < 1e-3, "triangle {i} was stressed by {s} under a rigid motion");
            }
            assert!(strain_energy(&m, &u, 210e9, 0.3).unwrap().abs() < 1e-6);
        }
    }

    #[test]
    fn an_underconstrained_body_is_reported_as_singular() {
        let m = FemMesh2::rect(1.0, 1.0, 3, 3).unwrap();
        let load = [(4usize, Vec2::new(1.0, 0.0))];
        // Nothing pinned.
        assert_eq!(
            fem_2d_elasticity_plane_stress(&m, 1.0, 0.3, &load, &[]),
            Err(SolveError::Singular)
        );
        // One node pinned kills both translations but leaves the
        // rotation about it free.
        assert_eq!(
            fem_2d_elasticity_plane_stress(&m, 1.0, 0.3, &load, &[(0, Vec2::ZERO)]),
            Err(SolveError::Singular)
        );
        // Two distinct nodes fix the rotation as well.
        assert!(fem_2d_elasticity_plane_stress(
            &m,
            1.0,
            0.3,
            &load,
            &[(0, Vec2::ZERO), (3, Vec2::ZERO)]
        )
        .is_ok());
    }

    #[test]
    fn a_uniform_strain_state_is_reproduced_exactly() {
        // The elasticity patch test: a linear displacement field is in
        // the element space, so prescribing it on the boundary must
        // reproduce it at the interior nodes and give a stress that is
        // the same on every triangle.
        let (e, nu) = (70e9, 0.33);
        let (ex, ey, gamma) = (1e-3, -4e-4, 6e-4);
        let field = move |p: Vec2| {
            Vec2::new(ex * p.x + 0.5 * gamma * p.y, 0.5 * gamma * p.x + ey * p.y)
        };
        let m = FemMesh2::rect(2.0, 1.5, 5, 4).unwrap();
        let u =
            fem_2d_elasticity_plane_stress(&m, e, nu, &[], &pin_boundary(&m, &field)).unwrap();
        // Relative to the displacement scale rather than absolute: the
        // displacements here are of order 1e-3, so an absolute
        // tolerance would be silently asking for three digits more than
        // a relative one.
        let scale = u.iter().fold(0.0f64, |a, d| a.max(d.x.abs()).max(d.y.abs()));
        for (i, got) in u.iter().enumerate() {
            let want = field(m.nodes[i]);
            let gap = (got.x - want.x).abs().max((got.y - want.y).abs());
            assert!(gap < 1e-13 * scale, "node {i} was off by {gap}, scale {scale}");
        }
        let d = plane_stress_d(e, nu);
        let want = [
            d[0][0] * ex + d[0][1] * ey,
            d[1][0] * ex + d[1][1] * ey,
            d[2][2] * gamma,
        ];
        for t in 0..m.tris.len() {
            let got = element_stress(&m, &u, e, nu, t).unwrap();
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-4 * want[k].abs().max(1.0),
                    "triangle {t} component {k}: {} vs {}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn the_closed_form_stress_states_come_out_right() {
        let (e, nu) = (200e9, 0.3);
        let m = FemMesh2::rect(1.0, 1.0, 3, 3).unwrap();
        // Uniaxial tension: strain (e0, -nu e0) gives sigma_x = E e0 and
        // sigma_y exactly zero, and a von Mises stress of |E e0|.
        let e0 = 2e-3;
        let uni = move |p: Vec2| Vec2::new(e0 * p.x, -nu * e0 * p.y);
        let u: Vec<Vec2> = m.nodes.iter().map(|&p| uni(p)).collect();
        let s = element_stress(&m, &u, e, nu, 0).unwrap();
        assert!((s[0] - e * e0).abs() < 1e-3 * e * e0);
        assert!(s[1].abs() < 1e-6 * e * e0, "the lateral stress was {}", s[1]);
        assert!((von_mises_stress(&m, &u, e, nu).unwrap()[0] - e * e0).abs() < 1e-3 * e * e0);
        // Pure shear: tau = G gamma with G = E / (2(1 + nu)), and the
        // von Mises stress of pure shear is sqrt(3) tau.
        let gamma = 1e-3;
        let sh: Vec<Vec2> =
            m.nodes.iter().map(|p| Vec2::new(0.5 * gamma * p.y, 0.5 * gamma * p.x)).collect();
        let g_mod = e / (2.0 * (1.0 + nu));
        let ss = element_stress(&m, &sh, e, nu, 0).unwrap();
        assert!((ss[2] - g_mod * gamma).abs() < 1e-4 * g_mod * gamma);
        assert!(ss[0].abs() < 1e-6 * g_mod * gamma && ss[1].abs() < 1e-6 * g_mod * gamma);
        let vm = von_mises_stress(&m, &sh, e, nu).unwrap()[0];
        assert!((vm - 3.0f64.sqrt() * g_mod * gamma).abs() < 1e-4 * vm);
        // Equal biaxial tension is *not* stress free in plane stress:
        // its von Mises value is the tension itself, because the free
        // surface makes the state anything but hydrostatic.
        let bi: Vec<Vec2> = m.nodes.iter().map(|p| Vec2::new(e0 * p.x, e0 * p.y)).collect();
        let bs = element_stress(&m, &bi, e, nu, 0).unwrap();
        assert!((bs[0] - bs[1]).abs() < 1e-6 * bs[0].abs());
        let bvm = von_mises_stress(&m, &bi, e, nu).unwrap()[0];
        assert!((bvm - bs[0].abs()).abs() < 1e-6 * bvm, "biaxial von Mises was {bvm}");
    }

    #[test]
    fn clapeyron_relates_the_work_to_the_strain_energy() {
        // At equilibrium the loads do exactly twice the stored strain
        // energy, which follows from the stiffness matrix being
        // symmetric and nothing else.
        let (e, nu) = (70e9, 0.3);
        let m = FemMesh2::rect(4.0, 1.0, 8, 2).unwrap();
        let clamped: Vec<(usize, Vec2)> = m
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x < 1e-12)
            .map(|(i, _)| (i, Vec2::ZERO))
            .collect();
        let loads: Vec<(usize, Vec2)> = m
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, p)| (p.x - 4.0).abs() < 1e-12)
            .map(|(i, _)| (i, Vec2::new(0.0, -1e6)))
            .collect();
        let u = fem_2d_elasticity_plane_stress(&m, e, nu, &loads, &clamped).unwrap();
        let work: f64 = loads.iter().map(|&(i, f)| f.x * u[i].x + f.y * u[i].y).sum();
        let energy = strain_energy(&m, &u, e, nu).unwrap();
        assert!(work > 0.0, "the tip did not move with the load");
        assert!((work - 2.0 * energy).abs() < 1e-6 * work, "{work} against {}", 2.0 * energy);
        // The cantilever deflects downwards and the tip moves most.
        assert!(u.iter().all(|d| d.y <= 1e-12));
        let tip = loads[0].0;
        assert!(u[tip].y < u[m.nodes.len() / 2].y);
        // Doubling the modulus halves the displacement, exactly.
        let stiffer = fem_2d_elasticity_plane_stress(&m, 2.0 * e, nu, &loads, &clamped).unwrap();
        assert!((stiffer[tip].y - 0.5 * u[tip].y).abs() < 1e-6 * u[tip].y.abs());
    }

    #[test]
    fn elasticity_refuses_impossible_materials_and_indices() {
        let m = FemMesh2::rect(1.0, 1.0, 2, 2).unwrap();
        let pin = [(0usize, Vec2::ZERO), (2, Vec2::ZERO)];
        assert!(fem_2d_elasticity_plane_stress(&m, -1.0, 0.3, &[], &pin).is_err());
        assert!(fem_2d_elasticity_plane_stress(&m, 1.0, 0.5, &[], &pin).is_err());
        assert!(fem_2d_elasticity_plane_stress(&m, 1.0, -1.0, &[], &pin).is_err());
        assert!(fem_2d_elasticity_plane_stress(&m, 1.0, 0.3, &[(99, Vec2::ZERO)], &pin).is_err());
        assert!(fem_2d_elasticity_plane_stress(&m, 1.0, 0.3, &[], &[(99, Vec2::ZERO)]).is_err());
        let u = vec![Vec2::ZERO; m.nodes.len()];
        assert!(von_mises_stress(&m, &u[..2], 1.0, 0.3).is_err());
        assert!(von_mises_stress(&m, &u, 1.0, 0.7).is_err());
        assert!(strain_energy(&m, &u[..2], 1.0, 0.3).is_err());
        assert!(element_strain(&m, &u, 9999).is_none());
        assert!(element_stress(&m, &u, 1.0, 0.7, 0).is_none());
    }

    #[test]
    fn an_insulated_body_conserves_its_heat_exactly() {
        // With no source and no prescribed boundary, the total heat
        // 1^T M u is unchanged by every step and for every theta,
        // because the stiffness rows sum to zero. This is exact, not
        // asymptotic: it is an algebraic consequence of the assembly.
        let m = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let initial: Vec<f64> =
            m.nodes.iter().map(|p| (3.0 * p.x).exp() * (1.0 + p.y)).collect();
        let mass = mass_matrix(&m);
        let total = |v: &[f64]| -> f64 { mass.mul_vec(v).iter().sum() };
        let start = total(&initial);
        for theta in [0.0, 0.5, 1.0] {
            let h = fem_2d_heat_transient(
                &m, &initial, 0.05, 0.01, 12, theta, &|_| 0.0, &|_| None,
            )
            .unwrap();
            assert_eq!(h.len(), 13);
            for (n, step) in h.iter().enumerate() {
                assert!(
                    (total(step) - start).abs() < 1e-9 * start.abs(),
                    "theta {theta} step {n} lost heat"
                );
            }
            // And it flattens: the spread between hottest and coldest
            // shrinks monotonically as diffusion does its work.
            let spread = |v: &[f64]| {
                v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    - v.iter().fold(f64::INFINITY, |a, &b| a.min(b))
            };
            assert!(spread(&h[12]) < spread(&h[0]));
        }
    }

    #[test]
    fn a_mode_decays_by_exactly_the_schemes_amplification_factor() {
        // Fed a discrete eigenmode, the theta scheme is a scalar
        // recurrence with factor (1 - (1-theta) a) / (1 + theta a),
        // a = alpha lambda dt. That identity is exact, so it separates a
        // time-stepping error from a spatial one completely.
        let m = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let (values, modes) = fem_eigenmodes_drum(&m, 1).unwrap();
        let (lambda, phi) = (values[0], &modes[0]);
        let (alpha, dt) = (0.3, 0.02);
        let a = alpha * lambda * dt;
        for theta in [0.0, 0.5, 1.0] {
            let factor = (1.0 - (1.0 - theta) * a) / (1.0 + theta * a);
            let h = fem_2d_heat_transient(
                &m, phi, alpha, dt, 6, theta, &|_| 0.0, &|_| Some(0.0),
            )
            .unwrap();
            let peak = phi.iter().fold(0.0f64, |x, &v| x.max(v.abs()));
            for (n, step) in h.iter().enumerate() {
                let want = factor.powi(n as i32);
                let got = step
                    .iter()
                    .zip(phi.iter())
                    .map(|(&s, &p)| if p.abs() > 0.5 * peak { s / p } else { want })
                    .fold(0.0f64, |x, r| x.max((r - want).abs()));
                assert!(got < 1e-7 * (1.0 + want.abs()), "theta {theta} step {n} drifted by {got}");
            }
        }
    }

    #[test]
    fn crank_nicolson_is_a_stable_without_being_l_stable() {
        // For a mode too stiff to resolve, backward Euler's factor tends
        // to zero and Crank-Nicolson's tends to minus one. So the stiff
        // mode dies under one scheme and survives, flipping sign every
        // step, under the other. Both are stable; only one is damping.
        let m = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let (values, modes) = fem_eigenmodes_drum(&m, 4).unwrap();
        let stiff_mode = &modes[3];
        let peak_at = stiff_mode
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        // A step far beyond what resolves the mode.
        let dt = 40.0 / values[3];
        let cn = fem_2d_heat_transient(
            &m, stiff_mode, 1.0, dt, 6, 0.5, &|_| 0.0, &|_| Some(0.0),
        )
        .unwrap();
        let be = fem_2d_heat_transient(
            &m, stiff_mode, 1.0, dt, 6, 1.0, &|_| 0.0, &|_| Some(0.0),
        )
        .unwrap();
        let start = stiff_mode[peak_at].abs();
        assert!(be[6][peak_at].abs() < 1e-4 * start, "backward Euler failed to damp");
        assert!(cn[6][peak_at].abs() > 0.5 * start, "Crank-Nicolson damped a stiff mode");
        // And it alternates sign, which is what a factor near -1 does.
        for n in 0..6 {
            assert!(
                cn[n][peak_at] * cn[n + 1][peak_at] < 0.0,
                "Crank-Nicolson did not oscillate at step {n}"
            );
        }
    }

    #[test]
    fn the_march_settles_onto_the_steady_solution() {
        let m = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let source = |p: Vec2| 1.0 + p.x;
        let hot = |p: Vec2| Some(0.2 * p.y);
        let steady = fem_2d_poisson(&m, &source, &hot).unwrap();
        let h = fem_2d_heat_transient(
            &m,
            &vec![0.0; m.nodes.len()],
            1.0,
            0.05,
            120,
            1.0,
            &source,
            &hot,
        )
        .unwrap();
        for i in 0..m.nodes.len() {
            assert!(
                (h[120][i] - steady[i]).abs() < 1e-6 * (1.0 + steady[i].abs()),
                "node {i}: {} against the steady {}",
                h[120][i],
                steady[i]
            );
        }
    }

    #[test]
    fn the_heat_march_refuses_impossible_arguments() {
        let m = FemMesh2::rect(1.0, 1.0, 2, 2).unwrap();
        let u0 = vec![0.0; m.nodes.len()];
        let no = |_: Vec2| None;
        assert!(fem_2d_heat_transient(&m, &u0[..2], 1.0, 0.1, 1, 1.0, &|_| 0.0, &no).is_err());
        assert!(fem_2d_heat_transient(&m, &u0, 1.0, 0.0, 1, 1.0, &|_| 0.0, &no).is_err());
        assert!(fem_2d_heat_transient(&m, &u0, 1.0, 0.1, 1, 1.5, &|_| 0.0, &no).is_err());
        assert!(fem_2d_heat_transient(&m, &u0, -1.0, 0.1, 1, 1.0, &|_| 0.0, &no).is_err());
        let bad = vec![f64::NAN; m.nodes.len()];
        assert!(fem_2d_heat_transient(&m, &bad, 1.0, 0.1, 1, 1.0, &|_| 0.0, &no).is_err());
    }

    #[test]
    fn a_delaunay_mesh_of_a_point_set_is_conforming() {
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                // A slight stagger keeps the point set from being
                // cocircular everywhere, which is the degenerate case
                // for a Delaunay triangulation.
                let s = if j % 2 == 0 { 0.0 } else { 0.13 };
                points.push(Vec2::new(i as f64 + s, j as f64));
            }
        }
        let m = FemMesh2::from_delaunay(&points).unwrap();
        assert!(m.tris.len() > 20);
        for t in &m.tris {
            assert!(signed_area(&m.nodes, t) > 0.0);
        }
        // The triangulation covers the convex hull, whose area is at
        // least that of the inner 4x4 block of the staggered grid.
        assert!(m.area() > 15.0, "the hull came out at {}", m.area());
    }
}
