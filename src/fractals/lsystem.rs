//! Lindenmayer systems: parallel string rewriting with simple,
//! stochastic, and context-sensitive rules, 2-D and 3-D turtle
//! interpretation of the ABOP alphabet (Prusinkiewicz & Lindenmayer,
//! "The Algorithmic Beauty of Plants", 1990), and a library of
//! classic presets. Parametric modules are out of scope: the rule
//! set here covers character rewriting only.

use crate::math::{Vec2, Vec3};
use crate::mesh::Mesh;
use crate::monte_carlo::Rng;
use crate::quaternion::Quaternion;
use crate::spatial::frame::Frame;
use crate::spatial::primitives::{Rect, Segment, Segment2};

/// A production rule.
#[derive(Debug, Clone)]
pub enum Rule {
    /// `from` rewrites to `to` unconditionally.
    Simple { from: char, to: String },
    /// `from` rewrites to one of the options, chosen with the given
    /// (relative) probabilities.
    Stochastic { from: char, options: Vec<(f64, String)> },
    /// `from` rewrites to `to` only when preceded by `left` and
    /// followed by `right` (either may be absent). Context matching
    /// skips characters in the system's `ignore` list and complete
    /// bracketed branches, as in ABOP's 1L/2L-system examples.
    Context { left: Option<String>, from: char, right: Option<String>, to: String },
}

/// A Lindenmayer system: axiom, production rules, the turtle turn
/// angle its drawings use (radians), and characters ignored during
/// context matching.
#[derive(Debug, Clone)]
pub struct LSystem {
    pub axiom: String,
    pub rules: Vec<Rule>,
    pub angle: f64,
    pub ignore: Vec<char>,
}

impl LSystem {
    /// New system with the given axiom and turtle angle in degrees.
    #[must_use]
    pub fn new(axiom: &str, angle_deg: f64) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: Vec::new(),
            angle: angle_deg.to_radians(),
            ignore: Vec::new(),
        }
    }

    /// Adds a simple rule (builder style).
    #[must_use]
    pub fn rule(mut self, from: char, to: &str) -> Self {
        self.rules.push(Rule::Simple { from, to: to.to_string() });
        self
    }

    /// Adds a stochastic rule with relative probabilities.
    ///
    /// # Panics
    /// Panics if no option has positive probability.
    #[must_use]
    pub fn stochastic_rule(mut self, from: char, options: &[(f64, &str)]) -> Self {
        assert!(
            options.iter().any(|&(p, _)| p > 0.0),
            "stochastic rule needs a positive probability"
        );
        self.rules.push(Rule::Stochastic {
            from,
            options: options.iter().map(|&(p, s)| (p, s.to_string())).collect(),
        });
        self
    }

    /// Scans left from `chars[i]` for the context string `ctx`
    /// (right-to-left), skipping ignored characters and complete
    /// bracketed branches.
    fn matches_left(&self, chars: &[char], i: usize, ctx: &str) -> bool {
        let want: Vec<char> = ctx.chars().collect();
        let mut w = want.len();
        let mut j = i;
        while w > 0 {
            if j == 0 {
                return false;
            }
            j -= 1;
            let c = chars[j];
            if c == ']' {
                // Skip the complete branch.
                let mut depth = 1;
                while depth > 0 {
                    if j == 0 {
                        return false;
                    }
                    j -= 1;
                    match chars[j] {
                        ']' => depth += 1,
                        '[' => depth -= 1,
                        _ => {}
                    }
                }
            } else if c == '[' || self.ignore.contains(&c) {
                // Branch openings and ignored symbols are transparent.
            } else {
                w -= 1;
                if want[w] != c {
                    return false;
                }
            }
        }
        true
    }

    /// Scans right from `chars[i]` for the context string `ctx`,
    /// skipping ignored characters and complete bracketed branches.
    fn matches_right(&self, chars: &[char], i: usize, ctx: &str) -> bool {
        let want: Vec<char> = ctx.chars().collect();
        let mut w = 0;
        let mut j = i + 1;
        while w < want.len() {
            if j >= chars.len() {
                return false;
            }
            let c = chars[j];
            if c == '[' {
                let mut depth = 1;
                j += 1;
                while depth > 0 {
                    if j >= chars.len() {
                        return false;
                    }
                    match chars[j] {
                        '[' => depth += 1,
                        ']' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                continue;
            }
            if c == ']' {
                return false; // end of this branch
            }
            if !self.ignore.contains(&c) {
                if want[w] != c {
                    return false;
                }
                w += 1;
            }
            j += 1;
        }
        true
    }

    fn rewrite_char(&self, chars: &[char], i: usize, rng: &mut Option<&mut Rng>) -> Option<String> {
        let c = chars[i];
        // Context rules first (most specific), then stochastic, then
        // simple; the first matching rule wins.
        for rule in &self.rules {
            if let Rule::Context { left, from, right, to } = rule {
                if *from == c
                    && left.as_ref().is_none_or(|l| self.matches_left(chars, i, l))
                    && right.as_ref().is_none_or(|r| self.matches_right(chars, i, r))
                {
                    return Some(to.clone());
                }
            }
        }
        for rule in &self.rules {
            match rule {
                Rule::Simple { from, to } if *from == c => return Some(to.clone()),
                Rule::Stochastic { from, options } if *from == c => {
                    let total: f64 = options.iter().map(|&(p, _)| p.max(0.0)).sum();
                    let mut pick = match rng {
                        Some(r) => r.next_f64() * total,
                        // Deterministic fallback: the first option.
                        None => 0.0,
                    };
                    for (p, s) in options {
                        pick -= p.max(0.0);
                        if pick <= 0.0 {
                            return Some(s.clone());
                        }
                    }
                    return Some(options.last().expect("non-empty options").1.clone());
                }
                _ => {}
            }
        }
        None
    }

    /// Rewrites the axiom `iterations` times (all characters in
    /// parallel per pass). `rng` drives stochastic rules; with `None`
    /// they deterministically pick their first option.
    #[must_use]
    pub fn generate(&self, iterations: usize, mut rng: Option<&mut Rng>) -> String {
        let mut current: Vec<char> = self.axiom.chars().collect();
        for _ in 0..iterations {
            let mut next = String::with_capacity(current.len() * 2);
            for i in 0..current.len() {
                match self.rewrite_char(&current, i, &mut rng) {
                    Some(s) => next.push_str(&s),
                    None => next.push(current[i]),
                }
            }
            current = next.chars().collect();
        }
        current.into_iter().collect()
    }
}

/// 2-D turtle interpreting the ABOP alphabet: `F`/`G` draw a step,
/// `f`/`g` move without drawing, `+`/`-` turn left/right by the
/// turn angle, `|` turns 180°, `[`/`]` push/pop state, `!` scales
/// the line width by `width_factor`. Other characters are ignored.
#[derive(Debug, Clone)]
pub struct Turtle2 {
    pub pos: Vec2,
    pub heading: f64,
    pub step: f64,
    pub angle: f64,
    pub pen_down: bool,
    pub line_width: f64,
    /// Multiplier applied to `line_width` by `!`.
    pub width_factor: f64,
    stack: Vec<(Vec2, f64, f64)>,
    lo: Vec2,
    hi: Vec2,
}

impl Turtle2 {
    /// Turtle at the origin heading +x.
    ///
    /// # Panics
    /// Panics unless `step > 0`.
    #[must_use]
    pub fn new(step: f64, angle_deg: f64) -> Self {
        assert!(step > 0.0, "turtle step must be positive");
        Self {
            pos: Vec2::ZERO,
            heading: 0.0,
            step,
            angle: angle_deg.to_radians(),
            pen_down: true,
            line_width: 1.0,
            width_factor: 0.7,
            stack: Vec::new(),
            lo: Vec2::ZERO,
            hi: Vec2::ZERO,
        }
    }

    fn touch(&mut self) {
        self.lo = Vec2::new(self.lo.x.min(self.pos.x), self.lo.y.min(self.pos.y));
        self.hi = Vec2::new(self.hi.x.max(self.pos.x), self.hi.y.max(self.pos.y));
    }

    /// Interprets the string, returning the drawn segments.
    pub fn interpret(&mut self, s: &str) -> Vec<Segment2> {
        self.interpret_with_width(s).into_iter().map(|(seg, _)| seg).collect()
    }

    /// Interprets the string, returning segments with the line width
    /// active while each was drawn.
    pub fn interpret_with_width(&mut self, s: &str) -> Vec<(Segment2, f64)> {
        let mut out = Vec::new();
        self.touch();
        for c in s.chars() {
            match c {
                'F' | 'G' => {
                    let from = self.pos;
                    self.pos = self.pos
                        + Vec2::new(self.heading.cos(), self.heading.sin()) * self.step;
                    self.touch();
                    if self.pen_down {
                        out.push((Segment2 { a: from, b: self.pos }, self.line_width));
                    }
                }
                'f' | 'g' => {
                    self.pos = self.pos
                        + Vec2::new(self.heading.cos(), self.heading.sin()) * self.step;
                    self.touch();
                }
                '+' => self.heading += self.angle,
                '-' => self.heading -= self.angle,
                '|' => self.heading += std::f64::consts::PI,
                '[' => self.stack.push((self.pos, self.heading, self.line_width)),
                ']' => {
                    if let Some((p, h, w)) = self.stack.pop() {
                        self.pos = p;
                        self.heading = h;
                        self.line_width = w;
                    }
                }
                '!' => self.line_width *= self.width_factor,
                _ => {}
            }
        }
        out
    }

    /// Bounding rectangle of every position visited so far.
    #[must_use]
    pub fn bounds(&self) -> Rect {
        Rect { min: self.lo, max: self.hi }
    }
}

/// 3-D turtle: the frame's local x axis is the heading, y the left
/// vector, z the up vector. `+`/`-` yaw about up, `&`/`^` pitch
/// about left, `\`/`/` roll about the heading, `|` yaws 180°,
/// `F` draws, `f` moves, `[`/`]` push/pop, `!` tapers the radius.
#[derive(Debug, Clone)]
pub struct Turtle3 {
    pub frame: Frame,
    pub step: f64,
    pub angle: f64,
    /// Current branch radius (for `interpret_tree` / `to_mesh`).
    pub radius: f64,
    /// Multiplier applied to `radius` by `!`.
    pub taper: f64,
    stack: Vec<(Frame, f64)>,
}

impl Turtle3 {
    /// Turtle at the origin heading +x with up +z.
    ///
    /// # Panics
    /// Panics unless `step > 0`.
    #[must_use]
    pub fn new(step: f64, angle_deg: f64) -> Self {
        assert!(step > 0.0, "turtle step must be positive");
        Self {
            frame: Frame::identity(),
            step,
            angle: angle_deg.to_radians(),
            radius: 1.0,
            taper: 0.7,
            stack: Vec::new(),
        }
    }

    fn turn(&mut self, local_axis: Vec3, angle: f64) {
        let axis = self.frame.to_world_vector(local_axis);
        let q = Quaternion::from_axis_angle(axis, angle);
        self.frame =
            Frame { origin: self.frame.origin, rotation: (q * self.frame.rotation).normalize() };
    }

    /// Interprets the string, returning drawn segments.
    pub fn interpret(&mut self, s: &str) -> Vec<Segment> {
        self.interpret_tree(s).into_iter().map(|(seg, _)| seg).collect()
    }

    /// Interprets the string, returning segments with the branch
    /// radius active while each was drawn.
    pub fn interpret_tree(&mut self, s: &str) -> Vec<(Segment, f64)> {
        let mut out = Vec::new();
        for c in s.chars() {
            match c {
                'F' | 'G' => {
                    let from = self.frame.origin;
                    let dir = self.frame.to_world_vector(Vec3::new(1.0, 0.0, 0.0));
                    self.frame.origin = from + dir * self.step;
                    out.push((Segment { a: from, b: self.frame.origin }, self.radius));
                }
                'f' | 'g' => {
                    let dir = self.frame.to_world_vector(Vec3::new(1.0, 0.0, 0.0));
                    self.frame.origin = self.frame.origin + dir * self.step;
                }
                '+' => self.turn(Vec3::new(0.0, 0.0, 1.0), self.angle),
                '-' => self.turn(Vec3::new(0.0, 0.0, 1.0), -self.angle),
                '&' => self.turn(Vec3::new(0.0, 1.0, 0.0), self.angle),
                '^' => self.turn(Vec3::new(0.0, 1.0, 0.0), -self.angle),
                '\\' => self.turn(Vec3::new(1.0, 0.0, 0.0), self.angle),
                '/' => self.turn(Vec3::new(1.0, 0.0, 0.0), -self.angle),
                '|' => self.turn(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::PI),
                '[' => self.stack.push((self.frame, self.radius)),
                ']' => {
                    if let Some((f, r)) = self.stack.pop() {
                        self.frame = f;
                        self.radius = r;
                    }
                }
                '!' => self.radius *= self.taper,
                _ => {}
            }
        }
        out
    }

    /// Interprets the string as a branching structure and meshes each
    /// drawn segment as an uncapped cylinder of its branch radius
    /// (starting from `base_radius`, multiplied by `taper` at each
    /// `!`).
    ///
    /// # Panics
    /// Panics unless `base_radius > 0` and `segments >= 3`.
    pub fn to_mesh(&mut self, s: &str, base_radius: f64, taper: f64, segments: usize) -> Mesh {
        assert!(base_radius > 0.0, "base radius must be positive");
        assert!(segments >= 3, "cylinder needs >= 3 segments");
        self.radius = base_radius;
        self.taper = taper;
        let branches = self.interpret_tree(s);
        let mut mesh = Mesh { vertices: Vec::new(), indices: Vec::new(), normals: None, uvs: None };
        for (seg, radius) in branches {
            let axis = seg.b - seg.a;
            let len = axis.magnitude();
            if len <= 0.0 {
                continue;
            }
            let mut cyl = crate::mesh::generate::cylinder(radius, len, segments, false);
            // The cylinder is y-aligned and centered; rotate +y onto
            // the branch axis and move it into place.
            let dir = axis * (1.0 / len);
            let y = Vec3::new(0.0, 1.0, 0.0);
            let d = y.dot(&dir).clamp(-1.0, 1.0);
            let q = if d > 1.0 - 1e-12 {
                Quaternion::identity()
            } else if d < -1.0 + 1e-12 {
                Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f64::consts::PI)
            } else {
                Quaternion::from_axis_angle(y.cross(&dir).normalized(), d.acos())
            };
            cyl.rotate(&q);
            cyl.translate((seg.a + seg.b) * 0.5);
            mesh.merge(&cyl);
        }
        mesh
    }
}

/// Classic L-systems, mostly from ABOP. Angles are the turtle turn
/// angles the figures were designed for.
pub mod presets {
    use super::LSystem;

    /// Koch curve: F → F+F−−F+F at 60°.
    #[must_use]
    pub fn koch_curve() -> LSystem {
        LSystem::new("F", 60.0).rule('F', "F+F--F+F")
    }

    /// Koch snowflake: the Koch rule on a triangle axiom.
    #[must_use]
    pub fn koch_snowflake() -> LSystem {
        LSystem::new("F--F--F", 60.0).rule('F', "F+F--F+F")
    }

    /// Quadratic Koch island (ABOP fig 1.7a): F → F+F−F−FF+F+F−F
    /// on a square, 90°.
    #[must_use]
    pub fn koch_island() -> LSystem {
        LSystem::new("F+F+F+F", 90.0).rule('F', "F+F-F-FF+F+F-F")
    }

    /// Heighway dragon at 90°.
    #[must_use]
    pub fn dragon() -> LSystem {
        LSystem::new("FX", 90.0).rule('X', "X+YF+").rule('Y', "-FX-Y")
    }

    /// Hilbert curve as an L-system (ABOP fig 1.11a), 90°.
    #[must_use]
    pub fn hilbert() -> LSystem {
        LSystem::new("A", 90.0).rule('A', "-BF+AFA+FB-").rule('B', "+AF-BFB-FA+")
    }

    /// Peano curve variant filling a square, 90°.
    #[must_use]
    pub fn peano() -> LSystem {
        LSystem::new("X", 90.0)
            .rule('X', "XFYFX+F+YFXFY-F-XFYFX")
            .rule('Y', "YFXFY-F-XFYFX+F+YFXFY")
    }

    /// Gosper flowsnake at 60° (F and G both draw).
    #[must_use]
    pub fn gosper() -> LSystem {
        LSystem::new("F", 60.0)
            .rule('F', "F-G--G+F++FF+G-")
            .rule('G', "+F-GG--G-F++F+G")
    }

    /// Sierpinski triangle (F and G both draw), 120°.
    #[must_use]
    pub fn sierpinski_triangle() -> LSystem {
        LSystem::new("F-G-G", 120.0).rule('F', "F-G+F+G-F").rule('G', "GG")
    }

    /// Sierpinski arrowhead curve, 60°.
    #[must_use]
    pub fn sierpinski_arrowhead() -> LSystem {
        LSystem::new("F", 60.0).rule('F', "G-F-G").rule('G', "F+G+F")
    }

    /// Lévy C curve, 45°.
    #[must_use]
    pub fn levy_c() -> LSystem {
        LSystem::new("F", 45.0).rule('F', "+F--F+")
    }

    /// Cantor set on a line: F draws, f skips the removed middle
    /// third.
    #[must_use]
    pub fn cantor() -> LSystem {
        LSystem::new("F", 0.0).rule('F', "FfF").rule('f', "fff")
    }

    /// ABOP fig 1.24a: F → F[+F]F[−F]F at 25.7°.
    #[must_use]
    pub fn plant_a() -> LSystem {
        LSystem::new("F", 25.7).rule('F', "F[+F]F[-F]F")
    }

    /// ABOP fig 1.24b: `F → F[+F]F[−F][F]` at 20°.
    #[must_use]
    pub fn plant_b() -> LSystem {
        LSystem::new("F", 20.0).rule('F', "F[+F]F[-F][F]")
    }

    /// ABOP fig 1.24c: F → FF−[−F+F+F]+[+F−F−F] at 22.5°.
    #[must_use]
    pub fn plant_c() -> LSystem {
        LSystem::new("F", 22.5).rule('F', "FF-[-F+F+F]+[+F-F-F]")
    }

    /// ABOP fig 1.24d: X → F[+X]F[−X]+X, F → FF at 20°.
    #[must_use]
    pub fn plant_d() -> LSystem {
        LSystem::new("X", 20.0).rule('X', "F[+X]F[-X]+X").rule('F', "FF")
    }

    /// ABOP fig 1.24e: `X → F[+X][−X]FX`, `F → FF` at 25.7°.
    #[must_use]
    pub fn plant_e() -> LSystem {
        LSystem::new("X", 25.7).rule('X', "F[+X][-X]FX").rule('F', "FF")
    }

    /// ABOP fig 1.24f: `X → F−[[X]+X]+F[+FX]−X`, `F → FF` at 22.5°.
    #[must_use]
    pub fn plant_f() -> LSystem {
        LSystem::new("X", 22.5).rule('X', "F-[[X]+X]+F[+FX]-X").rule('F', "FF")
    }

    /// Simple 3-D tree: trunk then three tapered branches rolled
    /// 120° apart (interpret with `Turtle3`).
    #[must_use]
    pub fn tree_3d() -> LSystem {
        LSystem::new("FA", 28.0)
            .rule('A', "!F[&FA]/[&FA]/[&FA]")
            .rule('/', "//") // widen the roll to ~120 deg per level
    }

    /// 3-D bush after ABOP fig 1.25 (interpret with `Turtle3`).
    #[must_use]
    pub fn bush_3d() -> LSystem {
        LSystem::new("A", 22.5)
            .rule('A', "[&FL!A]/////[&FL!A]///////[&FL!A]")
            .rule('F', "S/////F")
            .rule('S', "FL")
    }

    /// Cesàro curve: F → F+F−−F+F at 85°.
    #[must_use]
    pub fn cesaro() -> LSystem {
        LSystem::new("F", 85.0).rule('F', "F+F--F+F")
    }

    /// Pentaplexity (pentagonal flake curve), 36°.
    #[must_use]
    pub fn pentaplexity() -> LSystem {
        LSystem::new("F++F++F++F++F", 36.0).rule('F', "F++F++F|F-F++F")
    }

    /// Penrose P3 rhombus tiling as an L-system (the classic
    /// M/N/O/P system, angle 36°; draw F).
    #[must_use]
    pub fn penrose_lsystem() -> LSystem {
        LSystem::new("[N]++[N]++[N]++[N]++[N]", 36.0)
            .rule('M', "OF++PF----NF[-OF----MF]++")
            .rule('N', "+OF--PF[---MF--NF]+")
            .rule('O', "-MF++NF[+++OF++PF]-")
            .rule('P', "--OF++++MF[+PF++++NF]--NF")
            .rule('F', "")
    }

    /// Hexagonal Gosper curve (two-symbol XF form), 60°.
    #[must_use]
    pub fn hexagonal_gosper() -> LSystem {
        LSystem::new("XF", 60.0)
            .rule('X', "X+YF++YF-FX--FXFX-YF+")
            .rule('Y', "-FX+YFYF++YF+FX--FX-Y")
    }
}

/// Chains segments that share endpoints into polylines (in drawing
/// order): a new polyline starts whenever the pen jumped.
#[must_use]
pub fn lsystem_to_polylines(segments: &[Segment2]) -> Vec<Vec<Vec2>> {
    let mut out: Vec<Vec<Vec2>> = Vec::new();
    for seg in segments {
        if let Some(last) = out.last_mut() {
            let end = *last.last().expect("polylines are non-empty");
            if end.distance_to(&seg.a) < 1e-9 {
                last.push(seg.b);
                continue;
            }
        }
        out.push(vec![seg.a, seg.b]);
    }
    out
}

/// Box-counting dimension estimate of the drawing produced by
/// `iterations` rewrites. The two grid resolutions are chosen so the
/// finest cell is no smaller than a turtle step — below that scale
/// every curve is one-dimensional and the count slope collapses to 1.
///
/// # Panics
/// Panics if the drawing is empty or degenerate.
#[must_use]
pub fn fractal_dimension_lsystem(ls: &LSystem, iterations: usize) -> f64 {
    let s = ls.generate(iterations, None);
    let mut turtle = Turtle2::new(1.0, ls.angle.to_degrees());
    let segments = turtle.interpret(&s);
    assert!(!segments.is_empty(), "L-system draws nothing");
    let b = turtle.bounds();
    let extent = (b.max.x - b.min.x).max(b.max.y - b.min.y);
    assert!(extent > 0.0, "degenerate drawing");
    let bounds = (b.min.x, b.min.x + extent, b.min.y, b.min.y + extent);
    // Finest grid: cells no smaller than one turtle step (the
    // smallest self-similar feature), capped at 512 per side.
    let fine = usize::min(512, (extent.max(4.0) as usize).next_power_of_two());
    let cell = extent / fine as f64;
    let mut points = Vec::new();
    for seg in &segments {
        let len = seg.a.distance_to(&seg.b);
        let steps = (len / (0.5 * cell)).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = seg.a + (seg.b - seg.a) * t;
            points.push((p.x, p.y));
        }
    }
    let coarse = (fine / 4).max(2);
    let n1 = crate::fractals::box_count_2d(&points, coarse, bounds) as f64;
    let n2 = crate::fractals::box_count_2d(&points, fine, bounds) as f64;
    (n2 / n1).ln() / (fine as f64 / coarse as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_koch_growth() {
        let koch = presets::koch_curve();
        for n in 1..=5 {
            let s = koch.generate(n, None);
            let mut t = Turtle2::new(1.0, 60.0);
            let segs = t.interpret(&s);
            assert_eq!(segs.len(), 4usize.pow(n as u32), "Koch has 4^n segments");
            // All segments unit length; scaled to constant extent the
            // length grows by 4/3 per generation.
            for seg in &segs {
                assert!((seg.a.distance_to(&seg.b) - 1.0).abs() < 1e-9);
            }
            let b = t.bounds();
            let width = b.max.x - b.min.x;
            assert!((width - 3.0f64.powi(n as i32)).abs() < 1e-6, "Koch spans 3^n");
        }
    }

    #[test]
    fn test_hilbert_matches_space_filling() {
        for order in 2..=4u32 {
            let ls = presets::hilbert();
            let s = ls.generate(order as usize, None);
            let mut t = Turtle2::new(1.0, 90.0);
            let segs = t.interpret(&s);
            let n = 1u64 << order;
            assert_eq!(segs.len() as u64, n * n - 1, "Hilbert visits n^2 cells");
            // Unit steps everywhere -> a Hamiltonian path on the grid.
            for seg in &segs {
                assert!((seg.a.distance_to(&seg.b) - 1.0).abs() < 1e-9);
            }
            // Visited cells (normalized to grid indices) match the
            // cell set of space_filling::hilbert_curve_2d.
            let b = t.bounds();
            let mut cells: Vec<(i64, i64)> = Vec::with_capacity((n * n) as usize);
            let mut push = |p: crate::math::Vec2| {
                cells.push(((p.x - b.min.x).round() as i64, (p.y - b.min.y).round() as i64));
            };
            push(segs[0].a);
            for seg in &segs {
                push(seg.b);
            }
            cells.sort_unstable();
            cells.dedup();
            let reference: Vec<(i64, i64)> = {
                let mut v: Vec<(i64, i64)> =
                    crate::patterns::space_filling::hilbert_curve_2d(order)
                        .iter()
                        .map(|p| {
                            (
                                (p.x * n as f64 - 0.5).round() as i64,
                                (p.y * n as f64 - 0.5).round() as i64,
                            )
                        })
                        .collect();
                v.sort_unstable();
                v
            };
            assert_eq!(cells, reference, "same cell set as hilbert_curve_2d");
        }
    }

    #[test]
    fn test_dragon_no_self_intersection() {
        let s = presets::dragon().generate(10, None);
        let mut t = Turtle2::new(1.0, 90.0);
        let segs = t.interpret(&s);
        assert_eq!(segs.len(), 1024);
        for i in 0..segs.len() {
            for j in i + 1..segs.len() {
                let (a, b) = (&segs[i], &segs[j]);
                if let Some((s1, s2)) =
                    crate::spatial::intersect::segment_segment_2d_params(a, b)
                {
                    // Only endpoint contacts allowed.
                    let interior = s1 > 1e-9 && s1 < 1.0 - 1e-9 && s2 > 1e-9 && s2 < 1.0 - 1e-9;
                    assert!(!interior, "dragon self-intersects at pair ({i}, {j})");
                }
            }
        }
    }

    #[test]
    fn test_koch_box_dimension() {
        let d = fractal_dimension_lsystem(&presets::koch_curve(), 5);
        let expected = 4.0f64.ln() / 3.0f64.ln();
        assert!((d - expected).abs() < 0.05, "Koch dimension {d} vs {expected}");
    }

    #[test]
    fn test_stochastic_and_context_rules() {
        // Stochastic: both options appear over many runs.
        let ls = LSystem::new("F", 25.0)
            .stochastic_rule('F', &[(0.5, "F[+F]"), (0.5, "F[-F]")]);
        let mut rng = Rng::new(42);
        let mut saw_plus = false;
        let mut saw_minus = false;
        for _ in 0..40 {
            let s = ls.generate(1, Some(&mut rng));
            saw_plus |= s.contains("[+F]");
            saw_minus |= s.contains("[-F]");
        }
        assert!(saw_plus && saw_minus, "both stochastic options occur");
        // Deterministic fallback picks the first option.
        assert_eq!(ls.generate(1, None), "F[+F]");
        // Context: signal 'B' propagates right through 'A's (ABOP
        // 1L-system), skipping ignored '+'.
        let mut prop = LSystem::new("BAA+A", 0.0);
        prop.ignore.push('+');
        prop.rules.push(Rule::Context {
            left: Some("B".to_string()),
            from: 'A',
            right: None,
            to: "B".to_string(),
        });
        prop.rules.push(Rule::Simple { from: 'B', to: "A".to_string() });
        let s1 = prop.generate(1, None);
        assert_eq!(s1, "ABA+A", "signal moved one cell");
        let s2 = prop.generate(2, None);
        assert_eq!(s2, "AAB+A", "signal crossed the second cell");
        let s3 = prop.generate(3, None);
        assert_eq!(s3, "AAA+B", "signal skips the ignored symbol");
    }

    #[test]
    fn test_right_context_matching() {
        // A 1R-system: A → X only when followed by B.
        let mut ls = LSystem::new("ABAC", 0.0);
        ls.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("B".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(ls.generate(1, None), "XBAC", "only the A before B rewrites");
        // Multi-character right context.
        let mut two = LSystem::new("ABCABD", 0.0);
        two.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("BC".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(two.generate(1, None), "XBCABD");
        // Nothing to the right: no match at the end of the string.
        let mut edge = LSystem::new("A", 0.0);
        edge.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("B".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(edge.generate(1, None), "A", "no right neighbour, no match");

        // Ignored symbols are transparent to the right scan.
        let mut skip = LSystem::new("A+-BA+-C", 0.0);
        skip.ignore.extend(['+', '-']);
        skip.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("B".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(skip.generate(1, None), "X+-BA+-C");
        // Without the ignore list the same string does not match.
        let mut strict = skip.clone();
        strict.ignore.clear();
        assert_eq!(strict.generate(1, None), "A+-BA+-C");

        // Complete bracketed branches are skipped over, so the right
        // context sees the continuation of the current branch...
        let mut branch = LSystem::new("A[B]C", 0.0);
        branch.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("C".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(branch.generate(1, None), "X[B]C", "branch skipped");
        // ...and not the branch's own contents.
        let mut into = LSystem::new("A[B]C", 0.0);
        into.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("B".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(into.generate(1, None), "A[B]C", "branch contents are not context");
        // Nested branches are skipped as a unit.
        let mut nested = LSystem::new("A[B[C]D]E", 0.0);
        nested.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("E".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(nested.generate(1, None), "X[B[C]D]E");
        // An unclosed branch cannot be skipped: no match.
        let mut open = LSystem::new("A[BE", 0.0);
        open.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("E".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(open.generate(1, None), "A[BE");

        // A closing bracket ends the branch: nothing beyond it counts.
        let mut inside = LSystem::new("[A]B", 0.0);
        inside.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("B".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(inside.generate(1, None), "[A]B", "context stops at ']'");

        // Left and right context together (a 2L-system): the middle A
        // of "BAC" rewrites, the isolated ones do not.
        let mut both = LSystem::new("BACAAC", 0.0);
        both.rules.push(Rule::Context {
            left: Some("B".to_string()),
            from: 'A',
            right: Some("C".to_string()),
            to: "X".to_string(),
        });
        assert_eq!(both.generate(1, None), "BXCAAC");

        // A right-propagating signal: X moves leftward one cell per
        // step, the mirror image of the left-context example.
        let mut prop = LSystem::new("AAAX", 0.0);
        prop.rules.push(Rule::Context {
            left: None,
            from: 'A',
            right: Some("X".to_string()),
            to: "X".to_string(),
        });
        prop.rules.push(Rule::Simple { from: 'X', to: "A".to_string() });
        assert_eq!(prop.generate(1, None), "AAXA");
        assert_eq!(prop.generate(2, None), "AXAA");
        assert_eq!(prop.generate(3, None), "XAAA");
    }

    #[test]
    fn test_bush_3d_preset_expansion_and_drawing() {
        let bush = presets::bush_3d();
        assert_eq!(bush.axiom, "A");
        assert!((bush.angle - 22.5f64.to_radians()).abs() < 1e-15);
        let alphabet: Vec<char> = "AFLS&!/[]".chars().collect();
        let mut previous = bush.axiom.len();
        for n in 1..=5usize {
            let s = bush.generate(n, None);
            // Only alphabet symbols appear.
            for c in s.chars() {
                assert!(alphabet.contains(&c), "iteration {n}: stray symbol {c:?}");
            }
            // The expansion grows strictly with the iteration count.
            assert!(s.len() > previous, "iteration {n}: {} vs {previous}", s.len());
            previous = s.len();
            // A → three bracketed copies of itself: exactly 3^n apices.
            assert_eq!(
                s.chars().filter(|&c| c == 'A').count(),
                3usize.pow(n as u32),
                "iteration {n}: apex count"
            );
            // Brackets stay balanced and never go negative, and every
            // apex sits inside a branch.
            let mut depth = 0i32;
            let mut max_depth = 0i32;
            for c in s.chars() {
                match c {
                    '[' => {
                        depth += 1;
                        max_depth = max_depth.max(depth);
                    }
                    ']' => {
                        depth -= 1;
                        assert!(depth >= 0, "iteration {n}: unbalanced ']'");
                    }
                    _ => {}
                }
            }
            assert_eq!(depth, 0, "iteration {n}: unbalanced brackets");
            assert_eq!(max_depth, n as i32, "iteration {n}: nesting depth");
            // The 3-D turtle draws one segment per F, all of the turtle
            // step length, and returns to the origin at the end (every
            // branch is closed).
            let mut t = Turtle3::new(1.0, bush.angle.to_degrees());
            let segs = t.interpret(&s);
            assert_eq!(
                segs.len(),
                s.chars().filter(|&c| c == 'F').count(),
                "iteration {n}: one segment per F"
            );
            assert!(!segs.is_empty(), "iteration {n} draws nothing");
            for seg in &segs {
                assert!(
                    ((seg.b - seg.a).magnitude() - 1.0).abs() < 1e-12,
                    "iteration {n}: non-unit step"
                );
                assert!(seg.a.x.is_finite() && seg.b.z.is_finite());
            }
            assert!(
                t.frame.origin.magnitude() < 1e-12,
                "iteration {n}: turtle should return to the origin"
            );
            // The bush is genuinely three-dimensional: '&' pitches out
            // of the starting plane.
            if n >= 2 {
                let out_of_plane = segs.iter().any(|s| s.b.z.abs() > 0.1 || s.a.z.abs() > 0.1);
                assert!(out_of_plane, "iteration {n}: bush stayed planar");
            }
        }
        // Tapering: '!' shrinks the branch radius down the hierarchy.
        let mut t = Turtle3::new(1.0, 22.5);
        let mesh = t.to_mesh(&bush.generate(3, None), 0.05, 0.6, 6);
        assert!(!mesh.vertices.is_empty());
        assert!(mesh.surface_area() > 0.0);
        let radii: Vec<f64> = {
            let mut t2 = Turtle3::new(1.0, 22.5);
            t2.radius = 0.05;
            t2.taper = 0.6;
            t2.interpret_tree(&bush.generate(3, None)).into_iter().map(|(_, r)| r).collect()
        };
        let rmax = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rmin = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(rmin < rmax, "branches taper ({rmin} .. {rmax})");
        assert!((rmax - 0.05).abs() < 1e-12, "trunk keeps the base radius");
    }

    #[test]
    fn test_turtle3_and_mesh() {
        // A right angle in 3-D: F+F ends at (1, 1, 0) heading +y.
        let mut t = Turtle3::new(1.0, 90.0);
        let segs = t.interpret("F+F");
        assert_eq!(segs.len(), 2);
        assert!((segs[1].b - Vec3::new(1.0, 1.0, 0.0)).magnitude() < 1e-12);
        // Pitch down then draw: '&' rotates the heading toward -z...
        let mut t2 = Turtle3::new(1.0, 90.0);
        let segs2 = t2.interpret("&F");
        assert!(segs2[0].b.z.abs() > 0.99, "pitch moves out of the plane");
        // Bracket restore.
        let mut t3 = Turtle3::new(1.0, 90.0);
        let segs3 = t3.interpret("F[+F]F");
        assert!((segs3[2].b - Vec3::new(2.0, 0.0, 0.0)).magnitude() < 1e-12);
        // Tree mesh: manifold-ish output with positive area.
        let mut t4 = Turtle3::new(1.0, 28.0);
        let mesh = t4.to_mesh(&presets::tree_3d().generate(2, None), 0.1, 0.7, 6);
        assert!(!mesh.vertices.is_empty());
        assert!(mesh.surface_area() > 0.0);
    }

    #[test]
    fn test_polylines_and_presets_draw() {
        let mut t = Turtle2::new(1.0, 90.0);
        let segs = t.interpret("FF[+F]F");
        let polys = lsystem_to_polylines(&segs);
        // The branch causes one jump: F F +F | F -> two chains.
        assert_eq!(polys.len(), 2);
        assert_eq!(polys[0].len(), 4);
        for (name, ls, n) in [
            ("snowflake", presets::koch_snowflake(), 3),
            ("island", presets::koch_island(), 2),
            ("peano", presets::peano(), 2),
            ("gosper", presets::gosper(), 3),
            ("sierpinski", presets::sierpinski_triangle(), 4),
            ("arrowhead", presets::sierpinski_arrowhead(), 4),
            ("levy", presets::levy_c(), 6),
            ("cantor", presets::cantor(), 4),
            ("plant_a", presets::plant_a(), 3),
            ("plant_b", presets::plant_b(), 3),
            ("plant_c", presets::plant_c(), 3),
            ("plant_d", presets::plant_d(), 4),
            ("plant_e", presets::plant_e(), 4),
            ("plant_f", presets::plant_f(), 4),
            ("cesaro", presets::cesaro(), 4),
            ("pentaplexity", presets::pentaplexity(), 3),
            ("penrose", presets::penrose_lsystem(), 4),
            ("hex_gosper", presets::hexagonal_gosper(), 3),
        ] {
            let s = ls.generate(n, None);
            let mut turtle = Turtle2::new(1.0, ls.angle.to_degrees());
            let segs = turtle.interpret(&s);
            assert!(!segs.is_empty(), "{name} draws segments");
            for seg in &segs {
                assert!(seg.a.x.is_finite() && seg.b.y.is_finite(), "{name} finite");
            }
        }
    }

    #[test]
    fn test_sierpinski_triangle_dimension() {
        let d = fractal_dimension_lsystem(&presets::sierpinski_triangle(), 7);
        let expected = 3.0f64.ln() / 2.0f64.ln();
        assert!((d - expected).abs() < 0.1, "gasket dimension {d} vs {expected}");
    }
}
