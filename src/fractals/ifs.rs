//! Iterated function systems: the chaos game and deterministic
//! attractor construction (Barnsley, "Fractals Everywhere", 1988),
//! Moran similarity dimension, collage error, a library of classic
//! IFS presets in 2-D and 3-D, and Draves-style fractal flames.

use crate::math::{Vec2, Vec3};
use crate::monte_carlo::Rng;
use crate::spatial::mat4::Mat4;
use crate::spatial::primitives::{Polygon2, Rect};
use crate::spatial::transform2d::Affine2;

/// Builds the affine map x' = a·x + b·y + e, y' = c·x + d·y + f.
fn affine(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Affine2 {
    Affine2 { m: [[a, b, e], [c, d, f], [0.0, 0.0, 1.0]] }
}

/// A 2-D iterated function system: contractive affine maps with
/// selection probabilities.
#[derive(Debug, Clone)]
pub struct Ifs {
    pub maps: Vec<(Affine2, f64)>,
}

impl Ifs {
    /// New system; probabilities are normalized to sum to 1.
    ///
    /// # Panics
    /// Panics unless at least one map has positive probability.
    #[must_use]
    pub fn new(maps: Vec<(Affine2, f64)>) -> Self {
        let total: f64 = maps.iter().map(|&(_, p)| p.max(0.0)).sum();
        assert!(total > 0.0, "IFS needs a positive total probability");
        Self { maps: maps.into_iter().map(|(m, p)| (m, p.max(0.0) / total)).collect() }
    }

    fn pick(&self, u: f64) -> usize {
        let mut acc = 0.0;
        for (i, &(_, p)) in self.maps.iter().enumerate() {
            acc += p;
            if u < acc {
                return i;
            }
        }
        self.maps.len() - 1
    }

    /// The chaos game: iterate randomly chosen maps from the origin,
    /// discarding the first `burn_in` points, and return the next `n`
    /// points (which lie on the attractor to within the contraction
    /// tolerance).
    #[must_use]
    pub fn chaos_game(&self, n: usize, burn_in: usize, rng: &mut Rng) -> Vec<Vec2> {
        self.chaos_game_colored(n, burn_in, rng).into_iter().map(|(p, _)| p).collect()
    }

    /// Chaos game keeping the index of the map that produced each
    /// point (for per-map coloring).
    #[must_use]
    pub fn chaos_game_colored(
        &self,
        n: usize,
        burn_in: usize,
        rng: &mut Rng,
    ) -> Vec<(Vec2, usize)> {
        let mut p = Vec2::ZERO;
        let mut out = Vec::with_capacity(n);
        for i in 0..n + burn_in {
            let k = self.pick(rng.next_f64());
            p = self.maps[k].0.apply(p);
            if i >= burn_in {
                out.push((p, k));
            }
        }
        out
    }

    /// Deterministic construction: applies every map to every
    /// polygon, `depth` times, starting from `seed` — the m^depth
    /// results converge to the attractor in Hausdorff distance.
    ///
    /// # Panics
    /// Panics when m^depth would exceed 10^6 polygons.
    #[must_use]
    pub fn deterministic(&self, depth: usize, seed: &Polygon2) -> Vec<Polygon2> {
        assert!(
            (self.maps.len() as f64).powi(depth as i32) <= 1e6,
            "deterministic construction would produce too many polygons"
        );
        let mut current = vec![seed.clone()];
        for _ in 0..depth {
            let mut next = Vec::with_capacity(current.len() * self.maps.len());
            for poly in &current {
                for (map, _) in &self.maps {
                    next.push(Polygon2::new(
                        poly.vertices.iter().map(|&v| map.apply(v)).collect(),
                    ));
                }
            }
            current = next;
        }
        current
    }

    /// Deterministic point construction: all depth-fold compositions
    /// applied to the fixed point of the first map.
    ///
    /// # Panics
    /// Panics when m^depth would exceed 10^6 points.
    #[must_use]
    pub fn deterministic_points(&self, depth: usize) -> Vec<Vec2> {
        assert!(
            (self.maps.len() as f64).powi(depth as i32) <= 1e6,
            "deterministic construction would produce too many points"
        );
        // Fixed point of the first map: solve (I - A) x = t.
        let m = &self.maps[0].0.m;
        let (a, b, e) = (m[0][0], m[0][1], m[0][2]);
        let (c, d, f) = (m[1][0], m[1][1], m[1][2]);
        let det = (1.0 - a) * (1.0 - d) - b * c;
        let start = if det.abs() > 1e-12 {
            Vec2::new(((1.0 - d) * e + b * f) / det, (c * e + (1.0 - a) * f) / det)
        } else {
            Vec2::ZERO
        };
        let mut current = vec![start];
        for _ in 0..depth {
            let mut next = Vec::with_capacity(current.len() * self.maps.len());
            for &p in &current {
                for (map, _) in &self.maps {
                    next.push(map.apply(p));
                }
            }
            current = next;
        }
        current
    }

    /// Moran similarity dimension: the d solving Σ rᵢ^d = 1 where rᵢ
    /// are the contraction ratios, valid when every map is a
    /// similitude (uniform scale × rotation ± reflection) with
    /// ratio < 1. Returns `None` otherwise. Solved by bisection.
    #[must_use]
    pub fn similarity_dimension(&self) -> Option<f64> {
        let mut ratios = Vec::with_capacity(self.maps.len());
        for (map, _) in &self.maps {
            if !map.is_similarity(1e-9) {
                return None;
            }
            let det = map.m[0][0] * map.m[1][1] - map.m[0][1] * map.m[1][0];
            let r = det.abs().sqrt();
            if r <= 0.0 || r >= 1.0 {
                return None;
            }
            ratios.push(r);
        }
        let f = |d: f64| ratios.iter().map(|r| r.powf(d)).sum::<f64>() - 1.0;
        let (mut lo, mut hi) = (0.0, 10.0);
        if f(hi) > 0.0 {
            return None; // not contractive enough to pin down
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// Bounding rectangle of `n` chaos-game samples, padded by 1%.
    #[must_use]
    pub fn bounding_rect(&self, n: usize, rng: &mut Rng) -> Rect {
        let pts = self.chaos_game(n, 20, rng);
        let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &pts {
            lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
        let pad = (hi - lo) * 0.01;
        Rect { min: lo - pad, max: hi + pad }
    }

    /// Collage error: the symmetric Hausdorff distance between the
    /// target point set and the union of its images under the maps.
    /// The collage theorem bounds the distance from the target to the
    /// attractor by error/(1 − s) for contractivity s.
    ///
    /// # Panics
    /// Panics on an empty target.
    #[must_use]
    pub fn collage_error(&self, target: &[Vec2]) -> f64 {
        assert!(!target.is_empty(), "collage error needs target points");
        let images: Vec<Vec2> = target
            .iter()
            .flat_map(|&p| self.maps.iter().map(move |(m, _)| m.apply(p)))
            .collect();
        let directed = |from: &[Vec2], to: &[Vec2]| -> f64 {
            from.iter()
                .map(|p| {
                    to.iter()
                        .map(|q| p.distance_to(q))
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(0.0, f64::max)
        };
        directed(target, &images).max(directed(&images, target))
    }

    /// Renders `n` chaos-game samples into a `res.0` × `res.1` hit
    /// count grid (row-major, y up) over the attractor's bounding
    /// rectangle.
    ///
    /// # Panics
    /// Panics on a zero-sized grid.
    #[must_use]
    pub fn render_density(&self, n: usize, res: (usize, usize), rng: &mut Rng) -> Vec<u32> {
        assert!(res.0 > 0 && res.1 > 0, "render grid must be non-empty");
        let bounds = self.bounding_rect(2000.min(n.max(100)), rng);
        let size = bounds.max - bounds.min;
        let mut grid = vec![0u32; res.0 * res.1];
        for p in self.chaos_game(n, 20, rng) {
            let ix = ((p.x - bounds.min.x) / size.x * res.0 as f64) as usize;
            let iy = ((p.y - bounds.min.y) / size.y * res.1 as f64) as usize;
            if ix < res.0 && iy < res.1 {
                grid[iy * res.0 + ix] += 1;
            }
        }
        grid
    }
}

/// A 3-D IFS with affine maps stored as `Mat4`.
#[derive(Debug, Clone)]
pub struct Ifs3 {
    pub maps: Vec<(Mat4, f64)>,
}

impl Ifs3 {
    /// New system; probabilities are normalized to sum to 1.
    ///
    /// # Panics
    /// Panics unless at least one map has positive probability.
    #[must_use]
    pub fn new(maps: Vec<(Mat4, f64)>) -> Self {
        let total: f64 = maps.iter().map(|&(_, p)| p.max(0.0)).sum();
        assert!(total > 0.0, "IFS needs a positive total probability");
        Self { maps: maps.into_iter().map(|(m, p)| (m, p.max(0.0) / total)).collect() }
    }

    /// The chaos game in 3-D.
    #[must_use]
    pub fn chaos_game(&self, n: usize, burn_in: usize, rng: &mut Rng) -> Vec<Vec3> {
        let mut p = Vec3::ZERO;
        let mut out = Vec::with_capacity(n);
        for i in 0..n + burn_in {
            let u = rng.next_f64();
            let mut acc = 0.0;
            let mut k = self.maps.len() - 1;
            for (j, &(_, prob)) in self.maps.iter().enumerate() {
                acc += prob;
                if u < acc {
                    k = j;
                    break;
                }
            }
            p = self.maps[k].0.transform_point(p);
            if i >= burn_in {
                out.push(p);
            }
        }
        out
    }

    /// All depth-fold compositions applied to the origin.
    ///
    /// # Panics
    /// Panics when m^depth would exceed 10^6 points.
    #[must_use]
    pub fn deterministic_points(&self, depth: usize) -> Vec<Vec3> {
        assert!(
            (self.maps.len() as f64).powi(depth as i32) <= 1e6,
            "deterministic construction would produce too many points"
        );
        let mut current = vec![Vec3::ZERO];
        for _ in 0..depth {
            let mut next = Vec::with_capacity(current.len() * self.maps.len());
            for &p in &current {
                for (map, _) in &self.maps {
                    next.push(map.transform_point(p));
                }
            }
            current = next;
        }
        current
    }
}

/// Classic IFS attractors. 2-D maps are written x' = ax + by + e,
/// y' = cx + dy + f.
pub mod presets {
    use super::{affine, Affine2, Ifs, Ifs3};
    use crate::math::Vec3;
    use crate::spatial::mat4::Mat4;

    fn similitude(scale: f64, angle: f64, tx: f64, ty: f64) -> Affine2 {
        let (s, c) = angle.sin_cos();
        affine(scale * c, -scale * s, scale * s, scale * c, tx, ty)
    }

    fn scale3(s: f64, t: Vec3) -> Mat4 {
        Mat4::from_rows(
            [s, 0.0, 0.0, t.x],
            [0.0, s, 0.0, t.y],
            [0.0, 0.0, s, t.z],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Sierpinski triangle: three half-scale maps toward the corners
    /// of an equilateral triangle. Dimension log 3 / log 2.
    #[must_use]
    pub fn sierpinski() -> Ifs {
        let h = 3.0f64.sqrt() / 2.0;
        Ifs::new(vec![
            (similitude(0.5, 0.0, 0.0, 0.0), 1.0),
            (similitude(0.5, 0.0, 0.5, 0.0), 1.0),
            (similitude(0.5, 0.0, 0.25, 0.5 * h), 1.0),
        ])
    }

    /// Barnsley's fern (the classic four maps and probabilities).
    #[must_use]
    pub fn barnsley_fern() -> Ifs {
        Ifs::new(vec![
            (affine(0.0, 0.0, 0.0, 0.16, 0.0, 0.0), 0.01),
            (affine(0.85, 0.04, -0.04, 0.85, 0.0, 1.6), 0.85),
            (affine(0.2, -0.26, 0.23, 0.22, 0.0, 1.6), 0.07),
            (affine(-0.15, 0.28, 0.26, 0.24, 0.0, 0.44), 0.07),
        ])
    }

    /// Koch curve as four 1/3-scale similitudes. Dimension
    /// log 4 / log 3.
    #[must_use]
    pub fn koch() -> Ifs {
        let deg60 = std::f64::consts::FRAC_PI_3;
        Ifs::new(vec![
            (similitude(1.0 / 3.0, 0.0, 0.0, 0.0), 1.0),
            (similitude(1.0 / 3.0, deg60, 1.0 / 3.0, 0.0), 1.0),
            (similitude(1.0 / 3.0, -deg60, 0.5, 3.0f64.sqrt() / 6.0), 1.0),
            (similitude(1.0 / 3.0, 0.0, 2.0 / 3.0, 0.0), 1.0),
        ])
    }

    /// Heighway dragon: z → (1+i)z/2 and z → 1 − (1−i)z/2.
    #[must_use]
    pub fn dragon() -> Ifs {
        let s = 0.5f64.sqrt();
        let quarter = std::f64::consts::FRAC_PI_4;
        Ifs::new(vec![
            (similitude(s, quarter, 0.0, 0.0), 1.0),
            (similitude(s, 3.0 * quarter, 1.0, 0.0), 1.0),
        ])
    }

    /// Lévy C curve: z → wz and z → w̄z + (1 − w̄), w = (1+i)/2.
    #[must_use]
    pub fn levy() -> Ifs {
        let s = 0.5f64.sqrt();
        let quarter = std::f64::consts::FRAC_PI_4;
        Ifs::new(vec![
            (similitude(s, quarter, 0.0, 0.0), 1.0),
            (similitude(s, -quarter, 0.5, 0.5), 1.0),
        ])
    }

    /// Maple leaf (a well-known four-map collage).
    #[must_use]
    pub fn maple_leaf() -> Ifs {
        Ifs::new(vec![
            (affine(0.14, 0.01, 0.0, 0.51, -0.08, -1.31), 0.25),
            (affine(0.43, 0.52, -0.45, 0.5, 1.49, -0.75), 0.25),
            (affine(0.45, -0.49, 0.47, 0.47, -1.62, -0.74), 0.25),
            (affine(0.49, 0.0, 0.0, 0.51, 0.02, 1.62), 0.25),
        ])
    }

    /// Symmetric fractal tree: trunk, two rotated branches, and a
    /// crown copy.
    #[must_use]
    pub fn tree() -> Ifs {
        Ifs::new(vec![
            (affine(0.0, 0.0, 0.0, 0.5, 0.0, 0.0), 0.05),
            (affine(0.42, -0.42, 0.42, 0.42, 0.0, 0.2), 0.4),
            (affine(0.42, 0.42, -0.42, 0.42, 0.0, 0.2), 0.4),
            (affine(0.1, 0.0, 0.0, 0.1, 0.0, 0.2), 0.15),
        ])
    }

    /// Logarithmic spiral of copies: one strong rotation plus a
    /// small displaced copy.
    #[must_use]
    pub fn spiral() -> Ifs {
        Ifs::new(vec![
            (similitude(0.95, 0.3, 0.3, 0.0), 0.9),
            (similitude(0.15, 0.0, 1.0, 0.0), 0.1),
        ])
    }

    /// Cantor dust: four 1/3-scale copies at the unit square's
    /// corners. Dimension log 4 / log 3.
    #[must_use]
    pub fn cantor_dust() -> Ifs {
        let t = 2.0 / 3.0;
        Ifs::new(vec![
            (similitude(1.0 / 3.0, 0.0, 0.0, 0.0), 1.0),
            (similitude(1.0 / 3.0, 0.0, t, 0.0), 1.0),
            (similitude(1.0 / 3.0, 0.0, 0.0, t), 1.0),
            (similitude(1.0 / 3.0, 0.0, t, t), 1.0),
        ])
    }

    /// Pythagoras tree with roof angle `angle`: the two square-to-
    /// square similarities of the classic construction (the unit
    /// square is the trunk).
    ///
    /// # Panics
    /// Panics unless 0 < angle < π/2.
    #[must_use]
    pub fn pythagoras_tree(angle: f64) -> Ifs {
        assert!(
            angle > 0.0 && angle < std::f64::consts::FRAC_PI_2,
            "roof angle must be in (0, pi/2)"
        );
        let (s, c) = angle.sin_cos();
        Ifs::new(vec![
            (similitude(c, angle, 0.0, 1.0), 1.0),
            (similitude(s, angle - std::f64::consts::FRAC_PI_2, c * c, 1.0 + s * c), 1.0),
        ])
    }

    /// Sierpinski carpet: eight 1/3-scale copies (all but the
    /// center). Dimension log 8 / log 3.
    #[must_use]
    pub fn sierpinski_carpet() -> Ifs {
        let mut maps = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if i == 1 && j == 1 {
                    continue;
                }
                maps.push((
                    similitude(1.0 / 3.0, 0.0, i as f64 / 3.0, j as f64 / 3.0),
                    1.0,
                ));
            }
        }
        Ifs::new(maps)
    }

    /// Vicsek fractal (plus sign): center and four edge cells at
    /// 1/3 scale. Dimension log 5 / log 3.
    #[must_use]
    pub fn vicsek() -> Ifs {
        let t = 1.0 / 3.0;
        Ifs::new(vec![
            (similitude(t, 0.0, t, t), 1.0),
            (similitude(t, 0.0, 0.0, t), 1.0),
            (similitude(t, 0.0, 2.0 * t, t), 1.0),
            (similitude(t, 0.0, t, 0.0), 1.0),
            (similitude(t, 0.0, t, 2.0 * t), 1.0),
        ])
    }

    /// Menger sponge: the twenty 1/3-scale cells of the cube that
    /// survive (drop face centers and the body center). Dimension
    /// log 20 / log 3.
    #[must_use]
    pub fn menger_sponge_3d() -> Ifs3 {
        let mut maps = Vec::new();
        for i in 0..3i32 {
            for j in 0..3i32 {
                for k in 0..3i32 {
                    let ones = [i, j, k].iter().filter(|&&v| v == 1).count();
                    if ones >= 2 {
                        continue;
                    }
                    maps.push((
                        scale3(
                            1.0 / 3.0,
                            Vec3::new(
                                f64::from(i) / 3.0,
                                f64::from(j) / 3.0,
                                f64::from(k) / 3.0,
                            ),
                        ),
                        1.0,
                    ));
                }
            }
        }
        Ifs3::new(maps)
    }

    /// Sierpinski tetrahedron: four half-scale maps toward the
    /// vertices of a regular tetrahedron. Dimension 2.
    #[must_use]
    pub fn sierpinski_tetrahedron_3d() -> Ifs3 {
        let verts = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        Ifs3::new(verts.iter().map(|&v| (scale3(0.5, v * 0.5), 1.0)).collect())
    }

    /// Pentaflake: five copies at the vertices of a regular pentagon
    /// with contraction 1/(1+φ) = (3−√5)/2. Dimension
    /// log 5 / log(1+φ).
    #[must_use]
    pub fn pentagon_flake() -> Ifs {
        let r = (3.0 - 5.0f64.sqrt()) / 2.0;
        let maps = (0..5)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
                (similitude(r, 0.0, (1.0 - r) * a.cos(), (1.0 - r) * a.sin()), 1.0)
            })
            .collect();
        Ifs::new(maps)
    }

    /// Hexaflake: six vertex copies plus the center at 1/3 scale.
    /// Dimension log 7 / log 3.
    #[must_use]
    pub fn hexaflake() -> Ifs {
        let mut maps = vec![(similitude(1.0 / 3.0, 0.0, 0.0, 0.0), 1.0)];
        for i in 0..6 {
            let a = std::f64::consts::TAU * i as f64 / 6.0;
            maps.push((similitude(1.0 / 3.0, 0.0, 2.0 / 3.0 * a.cos(), 2.0 / 3.0 * a.sin()), 1.0));
        }
        Ifs::new(maps)
    }
}

/// The nonlinear variations of Draves & Reckase, "The Fractal Flame
/// Algorithm". Variations with free parameters use the fixed values
/// noted below; Julia uses the Ω = 0 branch so results are
/// deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variation {
    Linear,
    Sinusoidal,
    Spherical,
    Swirl,
    Horseshoe,
    Polar,
    Handkerchief,
    Heart,
    Disc,
    Spiral,
    Hyperbolic,
    Diamond,
    Ex,
    Julia,
    Bent,
    /// b = e = 0.5, c = f = 1.
    Waves,
    Fisheye,
    /// c = f = 0.1.
    Popcorn,
    Exponential,
    Power,
    Cosine,
    /// c² = 0.25.
    Rings,
    /// c² = 0.25, f = π/2.
    Fan,
}

/// Applies one flame variation to a point. Formulas follow the flame
/// paper's conventions: r = |p|, θ = atan2(x, y).
#[must_use]
pub fn apply_variation(v: Variation, p: Vec2) -> Vec2 {
    let (x, y) = (p.x, p.y);
    let r2 = x * x + y * y;
    let r = r2.sqrt();
    let theta = x.atan2(y);
    let pi = std::f64::consts::PI;
    match v {
        Variation::Linear => p,
        Variation::Sinusoidal => Vec2::new(x.sin(), y.sin()),
        Variation::Spherical => {
            let s = 1.0 / (r2 + 1e-12);
            Vec2::new(x * s, y * s)
        }
        Variation::Swirl => {
            let (s, c) = r2.sin_cos();
            Vec2::new(x * s - y * c, x * c + y * s)
        }
        Variation::Horseshoe => {
            let s = 1.0 / (r + 1e-12);
            Vec2::new(s * (x - y) * (x + y), s * 2.0 * x * y)
        }
        Variation::Polar => Vec2::new(theta / pi, r - 1.0),
        Variation::Handkerchief => Vec2::new(r * (theta + r).sin(), r * (theta - r).cos()),
        Variation::Heart => Vec2::new(r * (theta * r).sin(), -r * (theta * r).cos()),
        Variation::Disc => {
            let t = theta / pi;
            Vec2::new(t * (pi * r).sin(), t * (pi * r).cos())
        }
        Variation::Spiral => {
            let s = 1.0 / (r + 1e-12);
            Vec2::new(s * (theta.cos() + r.sin()), s * (theta.sin() - r.cos()))
        }
        Variation::Hyperbolic => Vec2::new(theta.sin() / (r + 1e-12), r * theta.cos()),
        Variation::Diamond => Vec2::new(theta.sin() * r.cos(), theta.cos() * r.sin()),
        Variation::Ex => {
            let p0 = (theta + r).sin();
            let p1 = (theta - r).cos();
            Vec2::new(r * (p0.powi(3) + p1.powi(3)), r * (p0.powi(3) - p1.powi(3)))
        }
        Variation::Julia => {
            let sq = r.sqrt();
            Vec2::new(sq * (0.5 * theta).cos(), sq * (0.5 * theta).sin())
        }
        Variation::Bent => Vec2::new(
            if x < 0.0 { 2.0 * x } else { x },
            if y < 0.0 { 0.5 * y } else { y },
        ),
        Variation::Waves => Vec2::new(x + 0.5 * y.sin(), y + 0.5 * x.sin()),
        Variation::Fisheye => {
            let s = 2.0 / (r + 1.0);
            Vec2::new(s * y, s * x)
        }
        Variation::Popcorn => Vec2::new(
            x + 0.1 * (3.0 * y).tan().sin(),
            y + 0.1 * (3.0 * x).tan().sin(),
        ),
        Variation::Exponential => {
            let e = (x - 1.0).exp();
            Vec2::new(e * (pi * y).cos(), e * (pi * y).sin())
        }
        Variation::Power => {
            let rp = r.powf(theta.sin());
            Vec2::new(rp * theta.cos(), rp * theta.sin())
        }
        Variation::Cosine => Vec2::new((pi * x).cos() * y.cosh(), -(pi * x).sin() * y.sinh()),
        Variation::Rings => {
            let c2 = 0.25;
            let m = ((r + c2).rem_euclid(2.0 * c2)) - c2 + r * (1.0 - c2);
            Vec2::new(m * theta.cos(), m * theta.sin())
        }
        Variation::Fan => {
            let t = pi * 0.25;
            let half = 0.5 * t;
            let f = std::f64::consts::FRAC_PI_2;
            if (theta + f).rem_euclid(t) > half {
                Vec2::new(r * (theta - half).cos(), r * (theta - half).sin())
            } else {
                Vec2::new(r * (theta + half).cos(), r * (theta + half).sin())
            }
        }
    }
}

/// Fractal flame chaos game: each step applies a randomly chosen
/// affine map followed by its variation, and blends a per-map color
/// coordinate c ← (c + cᵢ)/2 with cᵢ = i/(m−1). Returns points with
/// their color coordinates; non-finite excursions restart from the
/// origin.
///
/// # Panics
/// Panics on an empty map list or non-positive total probability.
#[must_use]
pub fn fractal_flame(
    maps: &[(Affine2, f64, Variation)],
    n: usize,
    rng: &mut Rng,
) -> Vec<(Vec2, f64)> {
    assert!(!maps.is_empty(), "flame needs at least one map");
    let total: f64 = maps.iter().map(|&(_, p, _)| p.max(0.0)).sum();
    assert!(total > 0.0, "flame needs a positive total probability");
    let mut p = Vec2::ZERO;
    let mut color = 0.5;
    let mut out = Vec::with_capacity(n);
    let burn_in = 20;
    for i in 0..n + burn_in {
        let mut u = rng.next_f64() * total;
        let mut k = maps.len() - 1;
        for (j, &(_, prob, _)) in maps.iter().enumerate() {
            u -= prob.max(0.0);
            if u <= 0.0 {
                k = j;
                break;
            }
        }
        let (ref m, _, v) = maps[k];
        p = apply_variation(v, m.apply(p));
        if !p.x.is_finite() || !p.y.is_finite() {
            p = Vec2::ZERO;
        }
        let target = if maps.len() > 1 { k as f64 / (maps.len() - 1) as f64 } else { 0.0 };
        color = 0.5 * (color + target);
        if i >= burn_in {
            out.push((p, color));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_dimensions() {
        let d = presets::sierpinski().similarity_dimension().expect("similitudes");
        assert!((d - 3.0f64.ln() / 2.0f64.ln()).abs() < 1e-9, "gasket dimension {d}");
        let d = presets::koch().similarity_dimension().expect("similitudes");
        assert!((d - 4.0f64.ln() / 3.0f64.ln()).abs() < 1e-9, "Koch dimension {d}");
        let d = presets::sierpinski_carpet().similarity_dimension().expect("similitudes");
        assert!((d - 8.0f64.ln() / 3.0f64.ln()).abs() < 1e-9, "carpet dimension {d}");
        let d = presets::vicsek().similarity_dimension().expect("similitudes");
        assert!((d - 5.0f64.ln() / 3.0f64.ln()).abs() < 1e-9, "Vicsek dimension {d}");
        let d = presets::dragon().similarity_dimension().expect("similitudes");
        assert!((d - 2.0).abs() < 1e-9, "dragon is plane-filling");
        // Pythagoras tree: cos^d + sin^d = 1 at 45 deg gives d = 2.
        let d = presets::pythagoras_tree(std::f64::consts::FRAC_PI_4)
            .similarity_dimension()
            .expect("similitudes");
        assert!((d - 2.0).abs() < 1e-9);
        // The fern's maps are not similitudes.
        assert!(presets::barnsley_fern().similarity_dimension().is_none());
    }

    #[test]
    fn test_chaos_game_in_bounds_and_deterministic_agreement() {
        let mut rng = Rng::new(7);
        let ifs = presets::sierpinski();
        let bounds = ifs.bounding_rect(4000, &mut rng);
        let pts = ifs.chaos_game(5000, 20, &mut rng);
        assert_eq!(pts.len(), 5000);
        // The rectangle is a sample estimate: fresh samples can peek
        // slightly past it near rarely-visited corners.
        let slack = 0.01 * (bounds.max - bounds.min).magnitude();
        for p in &pts {
            assert!(
                p.x >= bounds.min.x - slack
                    && p.x <= bounds.max.x + slack
                    && p.y >= bounds.min.y - slack
                    && p.y <= bounds.max.y + slack,
                "chaos game point {p:?} outside {bounds:?}"
            );
        }
        // Deterministic points converge to the same attractor: every
        // depth-8 point is near some chaos-game point.
        let det = ifs.deterministic_points(8);
        assert_eq!(det.len(), 3usize.pow(8));
        for q in det.iter().step_by(37) {
            let d = pts
                .iter()
                .map(|p| p.distance_to(q))
                .fold(f64::INFINITY, f64::min);
            assert!(d < 0.02, "deterministic point {q:?} far from attractor ({d})");
        }
    }

    #[test]
    fn test_legacy_generators_lie_on_attractor() {
        let mut rng = Rng::new(11);
        // sierpinski_point uses the unit triangle (0,0)(1,0)(0.5, √3/2)
        // -- the same attractor as presets::sierpinski().
        let samples = presets::sierpinski().chaos_game(100_000, 20, &mut rng);
        for &(x, y) in crate::fractals::sierpinski_point(0.25, 0.25, 60).iter().skip(10) {
            let p = Vec2::new(x, y);
            let d = samples
                .iter()
                .map(|q| q.distance_to(&p))
                .fold(f64::INFINITY, f64::min);
            assert!(d < 1e-2, "legacy Sierpinski point off the attractor ({d})");
        }
        let fern = presets::barnsley_fern().chaos_game(100_000, 20, &mut rng);
        for &(x, y) in crate::fractals::barnsley_fern_point(0.0, 0.0, 60).iter().skip(10) {
            let p = Vec2::new(x, y);
            let d = fern
                .iter()
                .map(|q| q.distance_to(&p))
                .fold(f64::INFINITY, f64::min);
            assert!(d < 2e-2, "legacy fern point off the attractor ({d})");
        }
    }

    #[test]
    fn test_deterministic_polygons_and_density() {
        let ifs = presets::sierpinski();
        let seed = Polygon2::new(vec![
            Vec2::ZERO,
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 3.0f64.sqrt() / 2.0),
        ]);
        let polys = ifs.deterministic(4, &seed);
        assert_eq!(polys.len(), 81);
        // Total area shrinks by (3/4) per level: 3 maps x (1/2)^2.
        let total: f64 = polys.iter().map(Polygon2::area).sum();
        assert!((total - seed.area() * 0.75f64.powi(4)).abs() < 1e-9);
        let mut rng = Rng::new(3);
        let grid = ifs.render_density(20_000, (32, 32), &mut rng);
        assert_eq!(grid.len(), 1024);
        let hits: u32 = grid.iter().sum();
        assert!(hits >= 19_000, "most samples land in the grid ({hits})");
        // Collage error of the true attractor is small.
        let mut rng2 = Rng::new(5);
        let target = ifs.chaos_game(2000, 20, &mut rng2);
        assert!(ifs.collage_error(&target) < 0.05);
    }

    #[test]
    fn test_ifs3_menger_and_tetrahedron() {
        let sponge = presets::menger_sponge_3d();
        assert_eq!(sponge.maps.len(), 20);
        let mut rng = Rng::new(9);
        let pts = sponge.chaos_game(2000, 20, &mut rng);
        for p in &pts {
            assert!(p.x >= -1e-9 && p.x <= 1.0 + 1e-9, "sponge stays in the unit cube");
            assert!(p.y >= -1e-9 && p.y <= 1.0 + 1e-9);
            assert!(p.z >= -1e-9 && p.z <= 1.0 + 1e-9);
            // No point in the (open) middle-third column through the
            // body center in any axis pair.
            let mid = |v: f64| v > 1.0 / 3.0 + 1e-9 && v < 2.0 / 3.0 - 1e-9;
            let middles = usize::from(mid(p.x)) + usize::from(mid(p.y)) + usize::from(mid(p.z));
            assert!(middles < 2, "chaos game point {p:?} in a removed cell");
        }
        let tetra = presets::sierpinski_tetrahedron_3d();
        let det = tetra.deterministic_points(6);
        assert_eq!(det.len(), 4096);
        for p in det.iter().step_by(41) {
            assert!(p.x.abs() <= 1.0 + 1e-9 && p.y.abs() <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_variations_and_flame() {
        let p = Vec2::new(0.3, -0.7);
        assert_eq!(apply_variation(Variation::Linear, p), p);
        let s = apply_variation(Variation::Sinusoidal, p);
        assert!((s.x - 0.3f64.sin()).abs() < 1e-15 && (s.y - (-0.7f64).sin()).abs() < 1e-15);
        // Spherical is an involution on the unit circle.
        let u = Vec2::new(0.6, 0.8);
        let sp = apply_variation(Variation::Spherical, u);
        assert!((sp - u).magnitude() < 1e-9);
        for v in [
            Variation::Swirl,
            Variation::Horseshoe,
            Variation::Polar,
            Variation::Handkerchief,
            Variation::Heart,
            Variation::Disc,
            Variation::Spiral,
            Variation::Hyperbolic,
            Variation::Diamond,
            Variation::Ex,
            Variation::Julia,
            Variation::Bent,
            Variation::Waves,
            Variation::Fisheye,
            Variation::Popcorn,
            Variation::Exponential,
            Variation::Power,
            Variation::Cosine,
            Variation::Rings,
            Variation::Fan,
        ] {
            let q = apply_variation(v, p);
            assert!(q.x.is_finite() && q.y.is_finite(), "{v:?} finite");
        }
        let mut rng = Rng::new(2);
        let maps = [
            (presets::sierpinski().maps[0].0, 1.0, Variation::Swirl),
            (presets::sierpinski().maps[1].0, 1.0, Variation::Sinusoidal),
            (presets::sierpinski().maps[2].0, 1.0, Variation::Spherical),
        ];
        let flame = fractal_flame(&maps, 5000, &mut rng);
        assert_eq!(flame.len(), 5000);
        for (q, c) in &flame {
            assert!(q.x.is_finite() && q.y.is_finite());
            assert!((0.0..=1.0).contains(c), "color coordinate in [0, 1]");
        }
    }

    #[test]
    fn test_remaining_presets_contract() {
        for (name, ifs) in [
            ("levy", presets::levy()),
            ("maple", presets::maple_leaf()),
            ("tree", presets::tree()),
            ("spiral", presets::spiral()),
            ("cantor", presets::cantor_dust()),
            ("pentaflake", presets::pentagon_flake()),
            ("hexaflake", presets::hexaflake()),
        ] {
            let mut rng = Rng::new(17);
            let pts = ifs.chaos_game(2000, 50, &mut rng);
            for p in &pts {
                assert!(p.x.is_finite() && p.y.is_finite(), "{name} stays finite");
                assert!(p.x.abs() < 100.0 && p.y.abs() < 100.0, "{name} bounded");
            }
        }
        // Pentaflake dimension log5 / log(1+phi).
        let d = presets::pentagon_flake().similarity_dimension().expect("similitudes");
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        assert!((d - 5.0f64.ln() / (1.0 + phi).ln()).abs() < 1e-9);
        // Probabilities normalize.
        let ifs = Ifs::new(vec![
            (Affine2::scaling(0.5, 0.5), 2.0),
            (Affine2::scaling(0.4, 0.4), 6.0),
        ]);
        assert!((ifs.maps[0].1 - 0.25).abs() < 1e-15);
        assert!((ifs.maps[1].1 - 0.75).abs() < 1e-15);
    }
}
