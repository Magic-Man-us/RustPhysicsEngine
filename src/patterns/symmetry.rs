//! Plane symmetry groups (the 17 wallpaper groups and 7 frieze
//! groups), lattices, 3-D point groups, symmetry detection, and
//! Hankin-style Islamic star patterns.
//!
//! Wallpaper operations are expressed in unit-cell (lattice)
//! coordinates: an element maps the unit cell to itself modulo unit
//! translations, so the returned sets are the coset representatives
//! of the point group (plus the centering translation for the
//! centered groups cm and cmm).

use crate::math::{Vec2, Vec3};
use crate::quaternion::Quaternion;
use crate::spatial::primitives::{Polygon2, Rect, Segment2};
use crate::spatial::Affine2;

/// The 17 wallpaper groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WallpaperGroup {
    P1,
    P2,
    Pm,
    Pg,
    Cm,
    Pmm,
    Pmg,
    Pgg,
    Cmm,
    P4,
    P4m,
    P4g,
    P3,
    P3m1,
    P31m,
    P6,
    P6m,
}

/// The 7 frieze groups (IUCr-style names).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FriezeGroup {
    /// Translations only (hop).
    P1,
    /// Glide reflection (step).
    P11g,
    /// Vertical mirrors (sidle).
    P1m1,
    /// Half turns (spinning hop).
    P2,
    /// Half turns + vertical mirrors + glide (spinning sidle).
    P2mg,
    /// Horizontal mirror (jump).
    P11m,
    /// Full symmetry (spinning jump).
    P2mm,
}

/// A 2-D lattice spanned by two basis vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lattice {
    pub a: Vec2,
    pub b: Vec2,
}

impl Lattice {
    #[must_use]
    pub fn square(s: f64) -> Self {
        Self { a: Vec2::new(s, 0.0), b: Vec2::new(0.0, s) }
    }

    /// Hexagonal lattice with 120° between the basis vectors.
    #[must_use]
    pub fn hexagonal(s: f64) -> Self {
        Self { a: Vec2::new(s, 0.0), b: Vec2::new(-0.5 * s, 3.0f64.sqrt() / 2.0 * s) }
    }

    #[must_use]
    pub fn rectangular(w: f64, h: f64) -> Self {
        Self { a: Vec2::new(w, 0.0), b: Vec2::new(0.0, h) }
    }

    /// Rhombic lattice: equal-length vectors at the given angle.
    #[must_use]
    pub fn rhombic(s: f64, angle: f64) -> Self {
        Self { a: Vec2::new(s, 0.0), b: Vec2::new(s * angle.cos(), s * angle.sin()) }
    }

    #[must_use]
    pub fn oblique(a: Vec2, b: Vec2) -> Self {
        Self { a, b }
    }

    /// Lagrange-Gauss reduction: returns an equivalent basis with the
    /// two shortest vectors (|a| <= |b|, |b| minimal).
    ///
    /// # Panics
    /// Panics for a degenerate (collinear) basis.
    #[must_use]
    pub fn reduce(&self) -> Lattice {
        let mut a = self.a;
        let mut b = self.b;
        assert!(a.cross(&b).abs() > 1e-12, "degenerate lattice basis");
        if a.magnitude_squared() > b.magnitude_squared() {
            std::mem::swap(&mut a, &mut b);
        }
        loop {
            let mu = (b.dot(&a) / a.magnitude_squared()).round();
            b = b - a * mu;
            if b.magnitude_squared() >= a.magnitude_squared() {
                break;
            }
            std::mem::swap(&mut a, &mut b);
        }
        Lattice { a, b }
    }

    /// World position of lattice coordinates (u, v).
    #[must_use]
    pub fn to_world(&self, p: Vec2) -> Vec2 {
        self.a * p.x + self.b * p.y
    }
}

fn affine(m: [[f64; 2]; 2], t: [f64; 2]) -> Affine2 {
    Affine2 {
        m: [
            [m[0][0], m[0][1], t[0]],
            [m[1][0], m[1][1], t[1]],
            [0.0, 0.0, 1.0],
        ],
    }
}

/// Canonical key: matrix entries plus translation reduced mod 1.
fn op_key(op: &Affine2) -> [i64; 6] {
    let quant = |x: f64| (x * 1e6).round() as i64;
    let modq = |x: f64| {
        let m = x.rem_euclid(1.0);
        let q = (m * 1e6).round() as i64;
        if q == 1_000_000 {
            0
        } else {
            q
        }
    };
    [
        quant(op.m[0][0]),
        quant(op.m[0][1]),
        quant(op.m[1][0]),
        quant(op.m[1][1]),
        modq(op.m[0][2]),
        modq(op.m[1][2]),
    ]
}

/// Closes a generator list under composition modulo unit
/// translations.
fn close_group(generators: &[Affine2]) -> Vec<Affine2> {
    let normalize = |op: &Affine2| {
        let mut m = op.m;
        m[0][2] = m[0][2].rem_euclid(1.0);
        m[1][2] = m[1][2].rem_euclid(1.0);
        if (m[0][2] - 1.0).abs() < 1e-9 {
            m[0][2] = 0.0;
        }
        if (m[1][2] - 1.0).abs() < 1e-9 {
            m[1][2] = 0.0;
        }
        Affine2 { m }
    };
    let mut ops = vec![Affine2::identity()];
    let mut keys = std::collections::HashSet::new();
    keys.insert(op_key(&ops[0]));
    let mut frontier = ops.clone();
    let mut guard = 0;
    while !frontier.is_empty() && guard < 100 {
        guard += 1;
        let mut next = Vec::new();
        for f in &frontier {
            for g in generators {
                let h = normalize(&g.compose(f));
                if keys.insert(op_key(&h)) {
                    ops.push(h);
                    next.push(h);
                }
            }
        }
        frontier = next;
    }
    ops
}

/// Generators of the group's point operations (with fractional
/// translations for glides and centering) in unit-cell coordinates.
fn wallpaper_gens(g: WallpaperGroup) -> Vec<Affine2> {
    use WallpaperGroup::*;
    let r2 = affine([[-1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]);
    let mx = affine([[-1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]); // mirror x -> -x
    let my = affine([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]);
    let center = affine([[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5]);
    let r4 = affine([[0.0, -1.0], [1.0, 0.0]], [0.0, 0.0]);
    // Hexagonal-cell operations (integer matrices in lattice coords).
    let r3 = affine([[0.0, -1.0], [1.0, -1.0]], [0.0, 0.0]);
    let r6 = affine([[1.0, -1.0], [1.0, 0.0]], [0.0, 0.0]);
    let m_swap = affine([[0.0, 1.0], [1.0, 0.0]], [0.0, 0.0]); // (x,y)->(y,x)
    let m_antiswap = affine([[0.0, -1.0], [-1.0, 0.0]], [0.0, 0.0]);
    match g {
        P1 => vec![],
        P2 => vec![r2],
        Pm => vec![mx],
        Pg => vec![affine([[-1.0, 0.0], [0.0, 1.0]], [0.0, 0.5])],
        Cm => vec![mx, center],
        Pmm => vec![mx, my],
        Pmg => vec![r2, affine([[-1.0, 0.0], [0.0, 1.0]], [0.5, 0.0])],
        Pgg => vec![r2, affine([[-1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])],
        Cmm => vec![mx, my, center],
        P4 => vec![r4],
        P4m => vec![r4, mx],
        P4g => vec![r4, affine([[-1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])],
        P3 => vec![r3],
        P3m1 => vec![r3, m_antiswap],
        P31m => vec![r3, m_swap],
        P6 => vec![r6],
        P6m => vec![r6, m_swap],
    }
}

/// The coset representatives of the wallpaper group's operations in
/// unit-cell coordinates (closed under composition modulo unit
/// translations; the centered groups include their centering
/// translation).
#[must_use]
pub fn wallpaper_generators(g: WallpaperGroup) -> Vec<Affine2> {
    close_group(&wallpaper_gens(g))
}

/// The order of the returned operation set.
#[must_use]
pub fn wallpaper_group_order(g: WallpaperGroup) -> usize {
    use WallpaperGroup::*;
    match g {
        P1 => 1,
        P2 | Pm | Pg => 2,
        Cm | Pmm | Pmg | Pgg | P4 => 4,
        Cmm | P4m | P4g => 8,
        P3 => 3,
        P3m1 | P31m | P6 => 6,
        P6m => 12,
    }
}

/// A natural lattice for the group at the given scale: square for the
/// tetragonal groups, hexagonal (120°) for the tri/hexagonal groups,
/// rectangular otherwise.
#[must_use]
pub fn wallpaper_lattice(g: WallpaperGroup, scale: f64) -> Lattice {
    use WallpaperGroup::*;
    match g {
        P4 | P4m | P4g => Lattice::square(scale),
        P3 | P3m1 | P31m | P6 | P6m => Lattice::hexagonal(scale),
        _ => Lattice::rectangular(scale, scale),
    }
}

/// A fundamental domain in unit-cell coordinates with area 1/order of
/// the cell. For the rectangular-cell groups it is a genuine
/// fundamental domain; for the centered and hexagonal groups it is an
/// area-correct representative slab (one valid choice among many
/// shapes).
#[must_use]
pub fn wallpaper_fundamental_domain(g: WallpaperGroup) -> Polygon2 {
    use WallpaperGroup::*;
    let quad = |a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)| {
        Polygon2::new(vec![
            Vec2::new(a.0, a.1),
            Vec2::new(b.0, b.1),
            Vec2::new(c.0, c.1),
            Vec2::new(d.0, d.1),
        ])
    };
    let tri = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        Polygon2::new(vec![Vec2::new(a.0, a.1), Vec2::new(b.0, b.1), Vec2::new(c.0, c.1)])
    };
    let slab = |order: f64| quad((0.0, 0.0), (1.0, 0.0), (1.0, 1.0 / order), (0.0, 1.0 / order));
    match g {
        P1 => quad((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        P2 | Pm | Pg => quad((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)),
        Cm | Pmm | Pmg | Pgg | P4 => quad((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
        Cmm | P4m | P4g => tri((0.0, 0.0), (0.5, 0.0), (0.5, 0.5)),
        P3 => slab(3.0),
        P3m1 | P31m | P6 => slab(6.0),
        P6m => slab(12.0),
    }
}

/// Tiles a motif (given in unit-cell coordinates) by the group and
/// lattice over the extent: every group operation applied to every
/// motif polygon, replicated over the lattice translations whose cell
/// origin falls in the extent.
#[must_use]
pub fn tile_motif(
    g: WallpaperGroup,
    motif: &[Polygon2],
    lattice: &Lattice,
    extent: &Rect,
) -> Vec<Polygon2> {
    let ops = wallpaper_generators(g);
    let mut out = Vec::new();
    for_each_cell(lattice, extent, |i, j| {
        for op in &ops {
            for poly in motif {
                let pts: Vec<Vec2> = poly
                    .vertices
                    .iter()
                    .map(|&p| {
                        let q = op.apply(p);
                        lattice.to_world(q + Vec2::new(f64::from(i), f64::from(j)))
                    })
                    .collect();
                out.push(Polygon2::new(pts));
            }
        }
    });
    out
}

/// Tiles a point set (unit-cell coordinates) by the group and lattice
/// over the extent.
#[must_use]
pub fn tile_points(
    g: WallpaperGroup,
    points: &[Vec2],
    lattice: &Lattice,
    extent: &Rect,
) -> Vec<Vec2> {
    let ops = wallpaper_generators(g);
    let mut out = Vec::new();
    for_each_cell(lattice, extent, |i, j| {
        for op in &ops {
            for &p in points {
                let q = op.apply(p);
                out.push(lattice.to_world(q + Vec2::new(f64::from(i), f64::from(j))));
            }
        }
    });
    out
}

fn for_each_cell(lattice: &Lattice, extent: &Rect, mut f: impl FnMut(i32, i32)) {
    let det = lattice.a.cross(&lattice.b);
    assert!(det.abs() > 1e-12, "degenerate lattice");
    let (mut lo_i, mut hi_i) = (i32::MAX, i32::MIN);
    let (mut lo_j, mut hi_j) = (i32::MAX, i32::MIN);
    for corner in extent.corners() {
        let u = corner.cross(&lattice.b) / det;
        let v = lattice.a.cross(&corner) / det;
        lo_i = lo_i.min(u.floor() as i32 - 1);
        hi_i = hi_i.max(u.ceil() as i32 + 1);
        lo_j = lo_j.min(v.floor() as i32 - 1);
        hi_j = hi_j.max(v.ceil() as i32 + 1);
    }
    for i in lo_i..=hi_i {
        for j in lo_j..=hi_j {
            let origin = lattice.to_world(Vec2::new(f64::from(i), f64::from(j)));
            if extent.contains_point(origin) {
                f(i, j);
            }
        }
    }
}

/// Frieze group operations modulo the period translation, in world
/// coordinates with the frieze axis along x and period `period`.
#[must_use]
pub fn frieze_generators(g: FriezeGroup, period: f64) -> Vec<Affine2> {
    use FriezeGroup::*;
    let e = Affine2::identity();
    let r2 = affine([[-1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]);
    let mv = affine([[-1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]); // vertical mirror
    let mh = affine([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]); // horizontal mirror
    let glide = affine([[1.0, 0.0], [0.0, -1.0]], [period / 2.0, 0.0]);
    match g {
        P1 => vec![e],
        P11g => vec![e, glide],
        P1m1 => vec![e, mv],
        P2 => vec![e, r2],
        P2mg => vec![e, r2, affine([[-1.0, 0.0], [0.0, 1.0]], [period / 2.0, 0.0]), glide],
        P11m => vec![e, mh],
        P2mm => vec![e, r2, mv, mh],
    }
}

/// Replicates a motif under the frieze group for `count` periods
/// (translations 0..count).
#[must_use]
pub fn frieze_motif(
    g: FriezeGroup,
    motif: &[Polygon2],
    period: f64,
    count: usize,
) -> Vec<Polygon2> {
    let ops = frieze_generators(g, period);
    let mut out = Vec::new();
    for k in 0..count {
        let shift = Vec2::new(period * k as f64, 0.0);
        for op in &ops {
            for poly in motif {
                out.push(Polygon2::new(
                    poly.vertices.iter().map(|&p| op.apply(p) + shift).collect(),
                ));
            }
        }
    }
    out
}

/// Rosette symmetry: the motif under the cyclic group C_n (rotations)
/// or dihedral D_n (`mirror` adds reflections), about the origin.
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn rosette(motif: &[Polygon2], n: u32, mirror: bool) -> Vec<Polygon2> {
    assert!(n >= 1, "rosette order must be >= 1");
    let mut out = Vec::new();
    for k in 0..n {
        let angle = 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(n);
        let rot = Affine2::rotation(angle);
        for poly in motif {
            out.push(Polygon2::new(poly.vertices.iter().map(|&p| rot.apply(p)).collect()));
            if mirror {
                let refl = rot.compose(&affine([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]));
                out.push(Polygon2::new(
                    poly.vertices.iter().map(|&p| refl.apply(p)).collect(),
                ));
            }
        }
    }
    out
}

/// Detects rotations and reflections about the centroid that map the
/// point set to itself within `tol`. Checks rotation orders up to the
/// point count and reflection axes through point/midpoint directions.
#[must_use]
pub fn detect_symmetries_2d(points: &[Vec2], tol: f64) -> Vec<Affine2> {
    if points.is_empty() {
        return vec![Affine2::identity()];
    }
    let centroid =
        points.iter().fold(Vec2::ZERO, |s, &p| s + p) * (1.0 / points.len() as f64);
    let maps_to_self = |op: &Affine2| {
        points.iter().all(|&p| {
            let q = op.apply(p - centroid) + centroid;
            points.iter().any(|&r| r.distance_to(&q) <= tol)
        })
    };
    let mut out = vec![Affine2::identity()];
    // Rotations by 2 pi / n, largest order first (its powers cover
    // the smaller compatible orders).
    for n in (2..=points.len().max(2)).rev() {
        let op = Affine2::rotation(2.0 * std::f64::consts::PI / n as f64);
        if maps_to_self(&op) {
            for k in 1..n {
                let r =
                    Affine2::rotation(2.0 * std::f64::consts::PI * k as f64 / n as f64);
                if out.iter().all(|o| op_key(&r) != op_key(o)) {
                    out.push(r);
                }
            }
            break;
        }
    }
    // Reflection axes through each point and each midpoint direction.
    let mut axes: Vec<f64> = Vec::new();
    for (i, &p) in points.iter().enumerate() {
        let d = p - centroid;
        if d.magnitude() > 1e-12 {
            axes.push(d.y.atan2(d.x));
        }
        for &q in points.iter().skip(i + 1) {
            let m = (p + q) * 0.5 - centroid;
            if m.magnitude() > 1e-12 {
                axes.push(m.y.atan2(m.x));
            }
        }
    }
    for angle in axes {
        let op = Affine2::rotation(angle)
            .compose(&affine([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0]))
            .compose(&Affine2::rotation(-angle));
        if maps_to_self(&op) && out.iter().all(|o| op_key(&op) != op_key(o)) {
            out.push(op);
        }
    }
    out
}

/// 3-D point groups (rotation parts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointGroup3 {
    /// Rotations of the tetrahedron (order 12).
    Tetrahedral,
    /// Rotations of the cube/octahedron (order 24).
    Octahedral,
    /// Rotations of the icosahedron/dodecahedron (order 60).
    Icosahedral,
    /// Cyclic rotations about z.
    Cn(u32),
    /// Dihedral: Cn plus n twofold axes perpendicular to z.
    Dn(u32),
    /// Cn with vertical mirrors (same rotations as Cn).
    Cnv(u32),
    /// Dn with horizontal mirror (same rotations as Dn).
    Dnh(u32),
}

fn quat_key(q: &Quaternion) -> [i64; 4] {
    // Canonicalize sign: q and -q are the same rotation.
    let mut v = [q.w, q.x, q.y, q.z];
    if v[0] < -1e-12
        || (v[0].abs() <= 1e-12
            && (v[1] < -1e-12 || (v[1].abs() <= 1e-12 && (v[2] < -1e-12 || (v[2].abs() <= 1e-12 && v[3] < 0.0)))))
    {
        for x in &mut v {
            *x = -*x;
        }
    }
    v.map(|x| (x * 1e6).round() as i64)
}

fn quat_close(generators: &[Quaternion]) -> Vec<Quaternion> {
    let mut ops = vec![Quaternion::identity()];
    let mut keys = std::collections::HashSet::new();
    keys.insert(quat_key(&ops[0]));
    let mut frontier = ops.clone();
    let mut guard = 0;
    while !frontier.is_empty() && guard < 40 {
        guard += 1;
        let mut next = Vec::new();
        for f in &frontier {
            for g in generators {
                let h = (*g * *f).normalize();
                if keys.insert(quat_key(&h)) {
                    ops.push(h);
                    next.push(h);
                }
            }
        }
        frontier = next;
    }
    ops
}

/// All proper rotations of the point group as quaternions (generated
/// by closure from the group's standard generators).
#[must_use]
pub fn point_group_rotations(g: PointGroup3) -> Vec<Quaternion> {
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    match g {
        PointGroup3::Tetrahedral => quat_close(&[
            Quaternion::from_axis_angle(Vec3::new(1.0, 1.0, 1.0), 2.0 * std::f64::consts::PI / 3.0),
            Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::PI),
        ]),
        PointGroup3::Octahedral => quat_close(&[
            Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2),
            Quaternion::from_axis_angle(Vec3::new(1.0, 1.0, 1.0), 2.0 * std::f64::consts::PI / 3.0),
        ]),
        PointGroup3::Icosahedral => quat_close(&[
            Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, phi), 2.0 * std::f64::consts::PI / 5.0),
            Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::PI),
        ]),
        PointGroup3::Cn(n) | PointGroup3::Cnv(n) => {
            assert!(n >= 1, "order must be >= 1");
            (0..n)
                .map(|k| {
                    Quaternion::from_axis_angle(
                        Vec3::new(0.0, 0.0, 1.0),
                        2.0 * std::f64::consts::PI * f64::from(k) / f64::from(n),
                    )
                })
                .collect()
        }
        PointGroup3::Dn(n) | PointGroup3::Dnh(n) => {
            assert!(n >= 1, "order must be >= 1");
            let mut out = Vec::with_capacity(2 * n as usize);
            for k in 0..n {
                let a = 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(n);
                out.push(Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), a));
                // Twofold axis in the xy plane at half the step angle.
                let axis = Vec3::new((a / 2.0).cos(), (a / 2.0).sin(), 0.0);
                out.push(Quaternion::from_axis_angle(axis, std::f64::consts::PI));
            }
            out
        }
    }
}

/// The full group order (including improper operations for the
/// mirror-bearing groups).
#[must_use]
pub fn point_group_order(g: PointGroup3) -> usize {
    match g {
        PointGroup3::Tetrahedral => 12,
        PointGroup3::Octahedral => 24,
        PointGroup3::Icosahedral => 60,
        PointGroup3::Cn(n) => n as usize,
        PointGroup3::Dn(n) | PointGroup3::Cnv(n) => 2 * n as usize,
        PointGroup3::Dnh(n) => 4 * n as usize,
    }
}

/// Orbit of a point under the group's rotations (deduplicated within
/// 1e-9 of the point scale).
#[must_use]
pub fn point_group_orbit(g: PointGroup3, p: Vec3) -> Vec<Vec3> {
    let tol = 1e-9 * p.magnitude().max(1.0);
    let mut out: Vec<Vec3> = Vec::new();
    for q in point_group_rotations(g) {
        let r = q.rotate_vec(p);
        if out.iter().all(|s| s.distance_to(&r) > tol) {
            out.push(r);
        }
    }
    out
}

/// Hankin's method for Islamic star patterns (after Kaplan): from two
/// points straddling each edge midpoint (offset `delta` along the
/// edge), rays leave into the polygon at `contact_angle` from the
/// edge; consecutive rays around the polygon are intersected to form
/// the strap segments.
///
/// # Panics
/// Panics unless `0 < contact_angle < π/2` and `delta >= 0`.
#[must_use]
pub fn hankin_star_pattern(
    tiling: &crate::patterns::tilings::Tiling,
    contact_angle: f64,
    delta: f64,
) -> Vec<Segment2> {
    assert!(
        contact_angle > 0.0 && contact_angle < std::f64::consts::FRAC_PI_2,
        "contact angle in (0, pi/2)"
    );
    assert!(delta >= 0.0, "delta must be nonnegative");
    let mut out = Vec::new();
    for face in &tiling.faces {
        let n = face.len();
        let pts: Vec<Vec2> = face.iter().map(|&v| tiling.vertices[v]).collect();
        // For each edge (interior on the left for counterclockwise
        // faces): one ray leaning toward the edge's start vertex, one
        // toward its end, launched from points straddling the
        // midpoint.
        struct EdgeRays {
            toward_start: (Vec2, Vec2),
            toward_end: (Vec2, Vec2),
        }
        let rays: Vec<EdgeRays> = (0..n)
            .map(|i| {
                let (a, b) = (pts[i], pts[(i + 1) % n]);
                let e = (b - a).normalized();
                let mid = (a + b) * 0.5;
                EdgeRays {
                    toward_start: (
                        mid + e * (delta / 2.0),
                        e.rotate(std::f64::consts::PI - contact_angle),
                    ),
                    toward_end: (mid - e * (delta / 2.0), e.rotate(contact_angle)),
                }
            })
            .collect();
        // The toward-end ray of edge i meets the toward-start ray of
        // edge i+1 near their shared vertex.
        for i in 0..n {
            let (p1, d1) = rays[i].toward_end;
            let (p2, d2) = rays[(i + 1) % n].toward_start;
            let denom = d1.cross(&d2);
            if denom.abs() < 1e-12 {
                continue;
            }
            let t = (p2 - p1).cross(&d2) / denom;
            let u = (p2 - p1).cross(&d1) / denom;
            if t <= 0.0 || u <= 0.0 {
                continue;
            }
            let x = p1 + d1 * t;
            out.push(Segment2 { a: p1, b: x });
            out.push(Segment2 { a: p2, b: x });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::tilings::{hexagonal_grid, Tiling};

    const ALL_WALLPAPER: [WallpaperGroup; 17] = [
        WallpaperGroup::P1,
        WallpaperGroup::P2,
        WallpaperGroup::Pm,
        WallpaperGroup::Pg,
        WallpaperGroup::Cm,
        WallpaperGroup::Pmm,
        WallpaperGroup::Pmg,
        WallpaperGroup::Pgg,
        WallpaperGroup::Cmm,
        WallpaperGroup::P4,
        WallpaperGroup::P4m,
        WallpaperGroup::P4g,
        WallpaperGroup::P3,
        WallpaperGroup::P3m1,
        WallpaperGroup::P31m,
        WallpaperGroup::P6,
        WallpaperGroup::P6m,
    ];

    #[test]
    fn test_wallpaper_groups_closed_with_expected_orders() {
        for g in ALL_WALLPAPER {
            let ops = wallpaper_generators(g);
            assert_eq!(
                ops.len(),
                wallpaper_group_order(g),
                "{g:?} order mismatch"
            );
            // Closure under composition modulo unit translations.
            for a in &ops {
                for b in &ops {
                    let c = a.compose(b);
                    let mut m = c.m;
                    m[0][2] = m[0][2].rem_euclid(1.0);
                    m[1][2] = m[1][2].rem_euclid(1.0);
                    let c = Affine2 { m };
                    assert!(
                        ops.iter().any(|o| op_key(o) == op_key(&c)),
                        "{g:?} not closed"
                    );
                }
            }
            // Fundamental domain area = cell area / order (unit cell).
            let dom = wallpaper_fundamental_domain(g);
            assert!(
                (dom.area() - 1.0 / ops.len() as f64).abs() < 1e-9,
                "{g:?} fundamental domain area {}",
                dom.area()
            );
        }
    }

    #[test]
    fn test_lattices() {
        let l = Lattice::oblique(Vec2::new(5.0, 0.1), Vec2::new(4.0, 0.3));
        let r = l.reduce();
        // Reduced basis spans the same lattice (equal cell area).
        assert!((l.a.cross(&l.b).abs() - r.a.cross(&r.b).abs()).abs() < 1e-9);
        assert!(r.a.magnitude() <= r.b.magnitude() + 1e-12);
        // Shortest vector of this lattice is (-1, 0.2).
        assert!(r.a.magnitude() < 1.1);
        let hexl = Lattice::hexagonal(2.0);
        assert!((hexl.a.magnitude() - 2.0).abs() < 1e-12);
        assert!((hexl.a.dot(&hexl.b) / (2.0 * 2.0) + 0.5).abs() < 1e-12, "120 degrees");
    }

    #[test]
    fn test_rectangular_and_rhombic_lattice_defining_relations() {
        // Rectangular: axis-aligned basis of the requested side
        // lengths, meeting at exactly 90°, cell area w·h.
        for &(w, h) in &[(1.0_f64, 1.0_f64), (3.0, 0.5), (0.25, 7.0)] {
            let l = Lattice::rectangular(w, h);
            assert_eq!(l.a, Vec2::new(w, 0.0));
            assert_eq!(l.b, Vec2::new(0.0, h));
            assert!((l.a.magnitude() - w).abs() < 1e-15);
            assert!((l.b.magnitude() - h).abs() < 1e-15);
            assert_eq!(l.a.dot(&l.b), 0.0, "rectangular basis is orthogonal");
            assert!((l.a.cross(&l.b).abs() - w * h).abs() < 1e-12, "cell area");
        }
        // A square is the rectangular lattice with equal sides.
        assert_eq!(Lattice::rectangular(2.0, 2.0), Lattice::square(2.0));

        // Rhombic: equal-length basis vectors separated by the given
        // angle, so cos θ = a·b/|a||b| and the area is s² sin θ.
        for &s in &[1.0_f64, 2.5] {
            for &deg in &[30.0_f64, 60.0, 72.0, 90.0, 120.0, 135.0] {
                let theta = deg.to_radians();
                let l = Lattice::rhombic(s, theta);
                assert!((l.a.magnitude() - s).abs() < 1e-12, "|a| at {deg} deg");
                assert!((l.b.magnitude() - s).abs() < 1e-12, "|b| at {deg} deg");
                let cos = l.a.dot(&l.b) / (l.a.magnitude() * l.b.magnitude());
                assert!((cos - theta.cos()).abs() < 1e-12, "angle at {deg} deg");
                assert!(
                    (l.a.cross(&l.b).abs() - s * s * theta.sin()).abs() < 1e-12,
                    "area at {deg} deg"
                );
                // The diagonals of a rhombus are perpendicular.
                let (p, q) = (l.a + l.b, l.a - l.b);
                assert!(p.dot(&q).abs() < 1e-12, "diagonals at {deg} deg");
            }
        }
        // Rhombic at 90° is the square lattice; at 120° it is the
        // hexagonal one.
        let sq = Lattice::rhombic(1.5, std::f64::consts::FRAC_PI_2);
        assert!((sq.b - Vec2::new(0.0, 1.5)).magnitude() < 1e-15);
        let hexl = Lattice::rhombic(2.0, 2.0 * std::f64::consts::FRAC_PI_3);
        let reference = Lattice::hexagonal(2.0);
        assert!((hexl.a - reference.a).magnitude() < 1e-12);
        assert!((hexl.b - reference.b).magnitude() < 1e-12);
    }

    #[test]
    fn test_wallpaper_lattice_matches_the_documented_cell_types() {
        use WallpaperGroup::*;
        let scale = 3.0_f64;
        for g in ALL_WALLPAPER {
            let l = wallpaper_lattice(g, scale);
            // The cell always has the requested edge length along a.
            assert!((l.a.magnitude() - scale).abs() < 1e-12, "{g:?} |a|");
            assert!((l.b.magnitude() - scale).abs() < 1e-12, "{g:?} |b|");
            let cos = l.a.dot(&l.b) / (scale * scale);
            match g {
                // Tetragonal groups get the square cell (90°).
                P4 | P4m | P4g => {
                    assert_eq!(l, Lattice::square(scale), "{g:?} is square");
                    assert!(cos.abs() < 1e-15);
                }
                // Tri/hexagonal groups get the 120° hexagonal cell.
                P3 | P3m1 | P31m | P6 | P6m => {
                    assert_eq!(l, Lattice::hexagonal(scale), "{g:?} is hexagonal");
                    assert!((cos + 0.5).abs() < 1e-12, "{g:?} 120 degrees");
                }
                // Everything else gets the rectangular cell.
                _ => {
                    assert_eq!(
                        l,
                        Lattice::rectangular(scale, scale),
                        "{g:?} is rectangular"
                    );
                    assert!(cos.abs() < 1e-15);
                }
            }
            // Every returned basis is non-degenerate and its cell area
            // scales as the square of the requested size.
            let area = l.a.cross(&l.b).abs();
            assert!(area > 0.0, "{g:?} degenerate");
            let doubled = wallpaper_lattice(g, 2.0 * scale);
            assert!(
                (doubled.a.cross(&doubled.b).abs() - 4.0 * area).abs() < 1e-9,
                "{g:?} area scaling"
            );
        }
        // The hexagonal cell has area (√3/2)·s², the square s².
        let h = wallpaper_lattice(P6m, 1.0);
        assert!((h.a.cross(&h.b).abs() - 3.0f64.sqrt() / 2.0).abs() < 1e-12);
        let s = wallpaper_lattice(P4m, 1.0);
        assert!((s.a.cross(&s.b).abs() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_tile_points_density() {
        // p4m applied to a generic point produces 8 images per cell.
        let lattice = Lattice::square(1.0);
        let extent = Rect { min: Vec2::new(-3.0, -3.0), max: Vec2::new(3.0, 3.0) };
        let pts = tile_points(
            WallpaperGroup::P4m,
            &[Vec2::new(0.13, 0.31)],
            &lattice,
            &extent,
        );
        // Cells with origin inside extent: 7x7 = 49 => 49 * 8 images.
        assert_eq!(pts.len(), 49 * 8);
        let motifs = tile_motif(
            WallpaperGroup::P2,
            &[Polygon2::new(vec![
                Vec2::new(0.1, 0.1),
                Vec2::new(0.3, 0.1),
                Vec2::new(0.2, 0.25),
            ])],
            &lattice,
            &extent,
        );
        assert_eq!(motifs.len(), 49 * 2);
        // p2 images have the same area (rigid maps).
        for m in &motifs {
            assert!((m.area() - 0.015).abs() < 1e-12);
        }
    }

    #[test]
    fn test_frieze_and_rosette() {
        for (g, order) in [
            (FriezeGroup::P1, 1),
            (FriezeGroup::P11g, 2),
            (FriezeGroup::P1m1, 2),
            (FriezeGroup::P2, 2),
            (FriezeGroup::P2mg, 4),
            (FriezeGroup::P11m, 2),
            (FriezeGroup::P2mm, 4),
        ] {
            assert_eq!(frieze_generators(g, 2.0).len(), order, "{g:?}");
        }
        let motif = [Polygon2::new(vec![
            Vec2::new(0.5, 0.1),
            Vec2::new(0.8, 0.1),
            Vec2::new(0.65, 0.4),
        ])];
        let strip = frieze_motif(FriezeGroup::P2mm, &motif, 2.0, 5);
        assert_eq!(strip.len(), 20);
        let ros = rosette(&motif, 6, false);
        assert_eq!(ros.len(), 6);
        let ros_d = rosette(&motif, 6, true);
        assert_eq!(ros_d.len(), 12);
        // Rotations preserve distance from origin.
        let r0 = motif[0].vertices[0].magnitude();
        for p in &ros {
            assert!((p.vertices[0].magnitude() - r0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_detect_symmetries() {
        // A perfect square: D4 = 8 symmetries.
        let square = [
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
        ];
        let syms = detect_symmetries_2d(&square, 1e-9);
        assert_eq!(syms.len(), 8, "square has D4 symmetry");
        // A scalene point set: only the identity.
        let scalene = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.3, 0.9)];
        assert_eq!(detect_symmetries_2d(&scalene, 1e-9).len(), 1);
        // Equilateral triangle: D3 = 6.
        let tri: Vec<Vec2> = (0..3)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
                Vec2::new(a.cos(), a.sin())
            })
            .collect();
        assert_eq!(detect_symmetries_2d(&tri, 1e-9).len(), 6);
    }

    #[test]
    fn test_point_groups() {
        for (g, rot_order) in [
            (PointGroup3::Tetrahedral, 12usize),
            (PointGroup3::Octahedral, 24),
            (PointGroup3::Icosahedral, 60),
            (PointGroup3::Cn(5), 5),
            (PointGroup3::Dn(4), 8),
        ] {
            let rots = point_group_rotations(g);
            assert_eq!(rots.len(), rot_order, "{g:?} rotation count");
            // Orbit size divides the rotation count.
            let orbit = point_group_orbit(g, Vec3::new(0.123, 0.456, 0.789));
            assert_eq!(rot_order % orbit.len(), 0, "{g:?} orbit size {}", orbit.len());
            // A generic point has a full orbit.
            assert_eq!(orbit.len(), rot_order, "{g:?} generic orbit");
        }
        // Full group orders.
        assert_eq!(point_group_order(PointGroup3::Cnv(6)), 12);
        assert_eq!(point_group_order(PointGroup3::Dnh(6)), 24);
        // Octahedral orbit of an axis point: the 6 face centers.
        let orbit = point_group_orbit(PointGroup3::Octahedral, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(orbit.len(), 6);
        // Icosahedral orbit of a 5-fold axis point: 12 vertices.
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        let orbit =
            point_group_orbit(PointGroup3::Icosahedral, Vec3::new(0.0, 1.0, phi).normalized());
        assert_eq!(orbit.len(), 12);
    }

    #[test]
    fn test_hankin_pattern() {
        let t: Tiling = hexagonal_grid(3, 3, 1.0, true);
        let segs = hankin_star_pattern(&t, std::f64::consts::FRAC_PI_4, 0.2);
        // Two strap segments per polygon edge.
        assert_eq!(segs.len(), t.faces.len() * 6 * 2);
        // All segments stay within the tiling's bounding box (plus
        // slack).
        let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for v in &t.vertices {
            lo = Vec2::new(lo.x.min(v.x), lo.y.min(v.y));
            hi = Vec2::new(hi.x.max(v.x), hi.y.max(v.y));
        }
        for s in &segs {
            for p in [s.a, s.b] {
                assert!(p.x >= lo.x - 1e-6 && p.x <= hi.x + 1e-6);
                assert!(p.y >= lo.y - 1e-6 && p.y <= hi.y + 1e-6);
            }
            assert!(s.a.distance_to(&s.b) > 1e-6, "nondegenerate strap");
        }
    }
}
