//! Cellular automata and growth models: elementary 1-D rules,
//! life-like 2-D automata with pattern/RLE placement, cyclic CA,
//! Langton's ant and turmites, Brian's Brain, Wireworld, 3-D
//! life-like rules, SmoothLife and Lenia (direct convolution),
//! abelian sandpiles, stochastic lattice models (forest fire,
//! Greenberg-Hastings, majority/voter dynamics, Schelling
//! segregation), reaction-diffusion systems (Gray-Scott,
//! Gierer-Meinhardt, FitzHugh-Nagumo, Oregonator, Brusselator), and
//! aggregation/percolation (DLA, Eden growth, invasion percolation).

use crate::error::GeomError;
use crate::math::Vec2;
use crate::monte_carlo::Rng;
use crate::spatial::primitives::Rect;

/// Elementary (radius-1, 2-state) 1-D cellular automaton with a
/// Wolfram rule number.
#[derive(Debug, Clone)]
pub struct Ca1D {
    pub rule: u8,
    pub cells: Vec<bool>,
    pub wrap: bool,
}

/// The rule as a lookup table indexed by the 3-bit neighborhood
/// (left·4 + center·2 + right).
#[must_use]
pub fn rule_table(rule: u8) -> [bool; 8] {
    let mut t = [false; 8];
    for (i, b) in t.iter_mut().enumerate() {
        *b = rule >> i & 1 == 1;
    }
    t
}

/// True for additive (XOR-linear) rules like 90, 150, 60: the rule
/// commutes with XOR of configurations.
#[must_use]
pub fn rule_is_additive(rule: u8) -> bool {
    let t = rule_table(rule);
    let f = |n: usize| u8::from(t[n]);
    if f(0) != 0 {
        return false; // linearity requires the zero neighborhood to map to 0
    }
    for a in 0..8usize {
        for b in 0..8usize {
            if f(a ^ b) != f(a) ^ f(b) {
                return false;
            }
        }
    }
    true
}

impl Ca1D {
    /// Automaton of `width` dead cells.
    ///
    /// # Panics
    /// Panics unless `width >= 3`.
    #[must_use]
    pub fn new(rule: u8, width: usize, wrap: bool) -> Self {
        assert!(width >= 3, "need at least 3 cells");
        Self { rule, cells: vec![false; width], wrap }
    }

    /// Sets the single center cell.
    pub fn seed_center(&mut self) {
        let n = self.cells.len();
        self.cells.fill(false);
        self.cells[n / 2] = true;
    }

    /// Sets each cell alive with probability `p`.
    pub fn seed_random(&mut self, rng: &mut Rng, p: f64) {
        for c in &mut self.cells {
            *c = rng.next_f64() < p;
        }
    }

    /// Advances one generation.
    pub fn step(&mut self) {
        let t = rule_table(self.rule);
        let n = self.cells.len();
        let get = |i: i64| -> bool {
            if self.wrap {
                self.cells[i.rem_euclid(n as i64) as usize]
            } else if i < 0 || i >= n as i64 {
                false
            } else {
                self.cells[i as usize]
            }
        };
        let next: Vec<bool> = (0..n as i64)
            .map(|i| {
                let idx = usize::from(get(i - 1)) << 2 | usize::from(get(i)) << 1
                    | usize::from(get(i + 1));
                t[idx]
            })
            .collect();
        self.cells = next;
    }

    /// Runs `steps` generations, returning every row including the
    /// initial state (steps+1 rows).
    pub fn run(&mut self, steps: usize) -> Vec<Vec<bool>> {
        let mut out = Vec::with_capacity(steps + 1);
        out.push(self.cells.clone());
        for _ in 0..steps {
            self.step();
            out.push(self.cells.clone());
        }
        out
    }

    /// Shannon entropy (bits) of the 3-cell block distribution of
    /// the current state.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        let n = self.cells.len();
        let mut counts = [0usize; 8];
        for i in 0..n {
            let idx = usize::from(self.cells[i]) << 2
                | usize::from(self.cells[(i + 1) % n]) << 1
                | usize::from(self.cells[(i + 2) % n]);
            counts[idx] += 1;
        }
        let total = n as f64;
        -counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total;
                p * p.log2()
            })
            .sum::<f64>()
    }

    /// Heuristic for Wolfram class 4 (complex localized structures).
    #[must_use]
    pub fn is_class4_heuristic(&self) -> bool {
        rule_classify_wolfram(self.rule) == 4
    }
}

/// Heuristic Wolfram classification of an elementary rule:
/// 1 = dies out, 2 = periodic/fixed, 3 = chaotic (high sustained
/// block entropy), 4 = complex (intermediate, long transients).
/// Based on entropy and activity statistics from a random seed; the
/// boundary between classes 3 and 4 is inherently fuzzy.
#[must_use]
pub fn rule_classify_wolfram(rule: u8) -> u8 {
    let width = 256;
    let mut ca = Ca1D::new(rule, width, true);
    let mut rng = Rng::new(12_345);
    ca.seed_random(&mut rng, 0.5);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        ca.step();
    }
    let mut activity = 0usize;
    let mut cycles = false;
    let mut prev = ca.cells.clone();
    for _ in 0..200 {
        ca.step();
        activity += ca.cells.iter().zip(&prev).filter(|(a, b)| a != b).count();
        prev = ca.cells.clone();
        if !seen.insert(ca.cells.clone()) {
            cycles = true;
            break;
        }
    }
    let alive = ca.cells.iter().filter(|&&c| c).count();
    if alive == 0 {
        return 1;
    }
    if cycles || activity == 0 {
        return 2;
    }
    let h = ca.entropy();
    let act = activity as f64 / (200.0 * width as f64);
    if h > 2.2 && act > 0.2 { 3 } else { 4 }
}

/// A life-like (outer-totalistic, Moore-neighborhood, 2-state)
/// automaton on a `w` × `h` grid.
#[derive(Debug, Clone)]
pub struct LifeLike {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<bool>,
    pub birth: [bool; 9],
    pub survive: [bool; 9],
    pub wrap: bool,
}

impl LifeLike {
    /// Parses a "B3/S23"-style rule string (also HighLife "B36/S23",
    /// Day & Night "B3678/S34678", Seeds "B2/S", Life without death
    /// "B3/S012345678").
    ///
    /// # Errors
    /// `InvalidArgument` on a malformed rule string.
    pub fn from_rule_string(w: usize, h: usize, rule: &str) -> Result<Self, GeomError> {
        let mut parts = rule.split('/');
        let (b, s) = match (parts.next(), parts.next(), parts.next()) {
            (Some(b), Some(s), None) => (b, s),
            _ => return Err(GeomError::InvalidArgument("rule must be B<digits>/S<digits>")),
        };
        if !b.starts_with(['B', 'b']) || !s.starts_with(['S', 's']) {
            return Err(GeomError::InvalidArgument("rule must be B<digits>/S<digits>"));
        }
        let mut birth = [false; 9];
        let mut survive = [false; 9];
        for (spec, table) in [(&b[1..], &mut birth), (&s[1..], &mut survive)] {
            for ch in spec.chars() {
                match ch.to_digit(10) {
                    Some(d) if d <= 8 => table[d as usize] = true,
                    _ => return Err(GeomError::InvalidArgument("rule digits must be 0-8")),
                }
            }
        }
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        Ok(Self { w, h, cells: vec![false; w * h], birth, survive, wrap: true })
    }

    /// Conway's Game of Life (B3/S23).
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3.
    #[must_use]
    pub fn conway(w: usize, h: usize) -> Self {
        Self::from_rule_string(w, h, "B3/S23").expect("valid rule")
    }

    fn live_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                let alive = if self.wrap {
                    self.cells[(ny.rem_euclid(self.h as i64) as usize) * self.w
                        + nx.rem_euclid(self.w as i64) as usize]
                } else if nx < 0 || ny < 0 || nx >= self.w as i64 || ny >= self.h as i64 {
                    false
                } else {
                    self.cells[ny as usize * self.w + nx as usize]
                };
                count += usize::from(alive);
            }
        }
        count
    }

    /// Advances one generation.
    pub fn step(&mut self) {
        let mut next = vec![false; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let n = self.live_neighbors(x, y);
                let alive = self.cells[y * self.w + x];
                next[y * self.w + x] = if alive { self.survive[n] } else { self.birth[n] };
            }
        }
        self.cells = next;
    }

    /// Advances `n` generations.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Number of live cells.
    #[must_use]
    pub fn population(&self) -> usize {
        self.cells.iter().filter(|&&c| c).count()
    }

    /// Stamps a `.O`-style pattern with its top-left corner at
    /// (x, y): 'O' or '*' set cells, everything else clears them.
    pub fn place(&mut self, x: usize, y: usize, pattern: &[&str]) {
        for (j, row) in pattern.iter().enumerate() {
            for (i, ch) in row.chars().enumerate() {
                let (cx, cy) = ((x + i) % self.w, (y + j) % self.h);
                self.cells[cy * self.w + cx] = ch == 'O' || ch == '*';
            }
        }
    }

    /// Stamps a run-length-encoded pattern (`bo$2bo$3o!` etc.):
    /// b = dead, o = alive, $ = next row, ! = end, digits repeat.
    ///
    /// # Errors
    /// `InvalidArgument` on unexpected characters.
    pub fn place_rle(&mut self, x: usize, y: usize, rle: &str) -> Result<(), GeomError> {
        let (mut cx, mut cy) = (x, y);
        let mut count = 0usize;
        for ch in rle.chars() {
            match ch {
                '0'..='9' => count = count * 10 + ch.to_digit(10).unwrap() as usize,
                'b' | 'o' => {
                    let n = count.max(1);
                    for _ in 0..n {
                        self.cells[(cy % self.h) * self.w + cx % self.w] = ch == 'o';
                        cx += 1;
                    }
                    count = 0;
                }
                '$' => {
                    cy += count.max(1);
                    cx = x;
                    count = 0;
                }
                '!' => return Ok(()),
                c if c.is_whitespace() => {}
                _ => return Err(GeomError::InvalidArgument("unexpected character in RLE")),
            }
        }
        Ok(())
    }

    /// Bounding rectangle of the live cells (cell centers), or
    /// `None` when empty.
    #[must_use]
    pub fn bounding_box(&self) -> Option<Rect> {
        let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut any = false;
        for y in 0..self.h {
            for x in 0..self.w {
                if self.cells[y * self.w + x] {
                    any = true;
                    lo = Vec2::new(lo.x.min(x as f64), lo.y.min(y as f64));
                    hi = Vec2::new(hi.x.max(x as f64), hi.y.max(y as f64));
                }
            }
        }
        any.then_some(Rect { min: lo, max: hi })
    }

    /// Steps a copy until the exact grid state recurs (oscillator or
    /// still-life period; spaceships on a wrapped grid recur when
    /// they lap the torus). `None` if no recurrence within
    /// `max_steps`.
    #[must_use]
    pub fn detect_period(&self, max_steps: usize) -> Option<usize> {
        let mut probe = self.clone();
        for k in 1..=max_steps {
            probe.step();
            if probe.cells == self.cells {
                return Some(k);
            }
        }
        None
    }

    /// True when one step leaves the grid unchanged.
    #[must_use]
    pub fn is_still_life(&self) -> bool {
        let mut probe = self.clone();
        probe.step();
        probe.cells == self.cells
    }

    /// Renders the grid as newline-separated `.O` rows.
    #[must_use]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity((self.w + 1) * self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                s.push(if self.cells[y * self.w + x] { 'O' } else { '.' });
            }
            s.push('\n');
        }
        s
    }
}

/// Classic Game of Life patterns in `.O` rows.
pub mod patterns {
    /// Glider (travels (1, 1) every 4 generations).
    #[must_use]
    pub fn glider() -> Vec<&'static str> {
        vec![".O.", "..O", "OOO"]
    }

    /// Lightweight spaceship (travels (2, 0) every 4 generations).
    #[must_use]
    pub fn lwss() -> Vec<&'static str> {
        vec![".O..O", "O....", "O...O", "OOOO."]
    }

    /// Gosper glider gun (period 30, emits one glider per period).
    #[must_use]
    pub fn gosper_gun() -> Vec<&'static str> {
        vec![
            "........................O...........",
            "......................O.O...........",
            "............OO......OO............OO",
            "...........O...O....OO............OO",
            "OO........O.....O...OO..............",
            "OO........O...O.OO....O.O...........",
            "..........O.....O.......O...........",
            "...........O...O....................",
            "............OO......................",
        ]
    }

    /// R-pentomino (long-lived methuselah).
    #[must_use]
    pub fn r_pentomino() -> Vec<&'static str> {
        vec![".OO", "OO.", ".O."]
    }

    /// Acorn (methuselah, stabilizes after 5206 generations).
    #[must_use]
    pub fn acorn() -> Vec<&'static str> {
        vec![".O.....", "...O...", "OO..OOO"]
    }

    /// Diehard (vanishes after 130 generations).
    #[must_use]
    pub fn diehard() -> Vec<&'static str> {
        vec!["......O.", "OO......", ".O...OOO"]
    }

    /// Pulsar (period-3 oscillator).
    #[must_use]
    pub fn pulsar() -> Vec<&'static str> {
        vec![
            "..OOO...OOO..",
            ".............",
            "O....O.O....O",
            "O....O.O....O",
            "O....O.O....O",
            "..OOO...OOO..",
            ".............",
            "..OOO...OOO..",
            "O....O.O....O",
            "O....O.O....O",
            "O....O.O....O",
            ".............",
            "..OOO...OOO..",
        ]
    }

    /// Pentadecathlon (period-15 oscillator).
    #[must_use]
    pub fn pentadecathlon() -> Vec<&'static str> {
        vec!["..O....O..", "OO.OOOO.OO", "..O....O.."]
    }

    /// Block (still life).
    #[must_use]
    pub fn block() -> Vec<&'static str> {
        vec!["OO", "OO"]
    }

    /// Beehive (still life).
    #[must_use]
    pub fn beehive() -> Vec<&'static str> {
        vec![".OO.", "O..O", ".OO."]
    }

    /// Blinker (period-2 oscillator).
    #[must_use]
    pub fn blinker() -> Vec<&'static str> {
        vec!["OOO"]
    }
}

/// Cyclic cellular automaton: state k advances to k+1 (mod states)
/// when at least `threshold` neighbors within Chebyshev `range`
/// carry the successor state; produces spiral waves.
#[derive(Debug, Clone)]
pub struct CyclicCa {
    pub w: usize,
    pub h: usize,
    pub states: u8,
    pub cells: Vec<u8>,
    pub threshold: usize,
    pub range: usize,
}

impl CyclicCa {
    /// Random initial configuration.
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3, `states >= 2`, and
    /// `range >= 1`.
    #[must_use]
    pub fn new(w: usize, h: usize, states: u8, threshold: usize, range: usize, rng: &mut Rng) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        assert!(states >= 2, "need at least two states");
        assert!(range >= 1, "range must be positive");
        let cells =
            (0..w * h).map(|_| (rng.next_f64() * f64::from(states)) as u8 % states).collect();
        Self { w, h, states, cells, threshold, range }
    }

    /// Advances one generation (toroidal).
    pub fn step(&mut self) {
        let mut next = self.cells.clone();
        let r = self.range as i64;
        for y in 0..self.h {
            for x in 0..self.w {
                let cur = self.cells[y * self.w + x];
                let succ = (cur + 1) % self.states;
                let mut count = 0usize;
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = (x as i64 + dx).rem_euclid(self.w as i64) as usize;
                        let ny = (y as i64 + dy).rem_euclid(self.h as i64) as usize;
                        count += usize::from(self.cells[ny * self.w + nx] == succ);
                    }
                }
                if count >= self.threshold {
                    next[y * self.w + x] = succ;
                }
            }
        }
        self.cells = next;
    }

    /// Advances `n` generations.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

/// Langton's ant generalized to multi-state turning rules ("RL" is
/// the classic ant; each letter gives the turn on a cell of that
/// color).
#[derive(Debug, Clone)]
pub struct LangtonsAnt {
    pub w: usize,
    pub h: usize,
    /// Cell colors 0..rule.len().
    pub cells: Vec<u8>,
    pub pos: (usize, usize),
    /// 0 = up, 1 = right, 2 = down, 3 = left.
    pub dir: u8,
    pub rule: String,
}

impl LangtonsAnt {
    /// Ant at the grid center on a blank toroidal grid.
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3 and the rule is made
    /// of L/R with at least 2 letters.
    #[must_use]
    pub fn new(w: usize, h: usize, rule: &str) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        assert!(
            rule.len() >= 2 && rule.chars().all(|c| c == 'L' || c == 'R'),
            "rule must be a string of L and R"
        );
        Self {
            w,
            h,
            cells: vec![0; w * h],
            pos: (w / 2, h / 2),
            dir: 0,
            rule: rule.to_string(),
        }
    }

    /// One ant step: turn by the current cell's rule letter, advance
    /// the cell color, move forward.
    pub fn step(&mut self) {
        let idx = self.pos.1 * self.w + self.pos.0;
        let color = self.cells[idx] as usize;
        let turn = self.rule.as_bytes()[color % self.rule.len()];
        self.dir = if turn == b'R' { (self.dir + 1) % 4 } else { (self.dir + 3) % 4 };
        self.cells[idx] = ((color + 1) % self.rule.len()) as u8;
        let (x, y) = (self.pos.0 as i64, self.pos.1 as i64);
        let (nx, ny) = match self.dir {
            0 => (x, y - 1),
            1 => (x + 1, y),
            2 => (x, y + 1),
            _ => (x - 1, y),
        };
        self.pos =
            (nx.rem_euclid(self.w as i64) as usize, ny.rem_euclid(self.h as i64) as usize);
    }

    /// Runs `n` steps.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Heuristic highway detection for the classic RL ant: the
    /// displacement over the last 104 steps repeats (the highway is
    /// a period-104 translation).
    #[must_use]
    pub fn highway_detected(&self) -> bool {
        let mut probe = self.clone();
        let start = probe.pos;
        probe.run(104);
        let mid = probe.pos;
        probe.run(104);
        let end = probe.pos;
        let d1 = (
            mid.0 as i64 - start.0 as i64,
            mid.1 as i64 - start.1 as i64,
        );
        let d2 = (end.0 as i64 - mid.0 as i64, end.1 as i64 - mid.1 as i64);
        d1 == d2 && d1 != (0, 0)
    }
}

/// A turmite: a two-dimensional Turing machine on cell colors. The
/// transition table maps (machine state, cell color) to (color to
/// write, turn in quarter-turns clockwise, next state).
#[derive(Debug, Clone)]
pub struct Turmite {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<u8>,
    pub pos: (usize, usize),
    pub dir: u8,
    pub state: u8,
    /// Indexed by [state][color].
    pub table: Vec<Vec<(u8, u8, u8)>>,
}

impl Turmite {
    /// Turmite at the center of a blank toroidal grid.
    ///
    /// # Panics
    /// Panics on an empty transition table or a grid under 3×3.
    #[must_use]
    pub fn new(w: usize, h: usize, table: Vec<Vec<(u8, u8, u8)>>) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        assert!(!table.is_empty() && !table[0].is_empty(), "empty transition table");
        Self { w, h, cells: vec![0; w * h], pos: (w / 2, h / 2), dir: 0, state: 0, table }
    }

    /// One machine step.
    pub fn step(&mut self) {
        let idx = self.pos.1 * self.w + self.pos.0;
        let color = self.cells[idx] as usize % self.table[0].len();
        let (write, turn, next) = self.table[self.state as usize % self.table.len()][color];
        self.cells[idx] = write;
        self.dir = (self.dir + turn) % 4;
        self.state = next;
        let (x, y) = (self.pos.0 as i64, self.pos.1 as i64);
        let (nx, ny) = match self.dir {
            0 => (x, y - 1),
            1 => (x + 1, y),
            2 => (x, y + 1),
            _ => (x - 1, y),
        };
        self.pos =
            (nx.rem_euclid(self.w as i64) as usize, ny.rem_euclid(self.h as i64) as usize);
    }

    /// Runs `n` steps.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

/// Brian's Brain: three states (0 dead, 1 dying, 2 firing); a dead
/// cell fires with exactly two firing neighbors, firing cells decay
/// to dying, dying cells die.
#[derive(Debug, Clone)]
pub struct BriansBrain {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<u8>,
}

impl BriansBrain {
    /// Blank grid.
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3.
    #[must_use]
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        Self { w, h, cells: vec![0; w * h] }
    }

    /// Advances one generation (toroidal).
    pub fn step(&mut self) {
        let mut next = vec![0u8; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let cur = self.cells[y * self.w + x];
                next[y * self.w + x] = match cur {
                    2 => 1,
                    1 => 0,
                    _ => {
                        let mut firing = 0;
                        for dy in -1i64..=1 {
                            for dx in -1i64..=1 {
                                if dx == 0 && dy == 0 {
                                    continue;
                                }
                                let nx = (x as i64 + dx).rem_euclid(self.w as i64) as usize;
                                let ny = (y as i64 + dy).rem_euclid(self.h as i64) as usize;
                                firing += usize::from(self.cells[ny * self.w + nx] == 2);
                            }
                        }
                        u8::from(firing == 2) * 2
                    }
                };
            }
        }
        self.cells = next;
    }
}

/// Wireworld: 0 empty, 1 electron head, 2 electron tail,
/// 3 conductor. Heads become tails, tails become conductor, and a
/// conductor becomes a head with one or two neighboring heads.
#[derive(Debug, Clone)]
pub struct Wireworld {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<u8>,
}

impl Wireworld {
    /// Parses a circuit: ' ' or '.' empty, 'C' or '#' conductor,
    /// 'H' electron head, 'T' electron tail. Rows are padded to the
    /// longest line.
    ///
    /// # Errors
    /// `InvalidArgument` on unknown characters or an empty diagram.
    pub fn from_string(diagram: &str) -> Result<Self, GeomError> {
        let lines: Vec<&str> = diagram.lines().collect();
        let h = lines.len();
        let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
        if w < 1 || h < 1 {
            return Err(GeomError::InvalidArgument("empty Wireworld diagram"));
        }
        let mut cells = vec![0u8; w * h];
        for (y, line) in lines.iter().enumerate() {
            for (x, ch) in line.chars().enumerate() {
                cells[y * w + x] = match ch {
                    ' ' | '.' => 0,
                    'H' | 'h' => 1,
                    'T' | 't' => 2,
                    'C' | '#' => 3,
                    _ => return Err(GeomError::InvalidArgument("unknown Wireworld character")),
                };
            }
        }
        Ok(Self { w, h, cells })
    }

    /// Advances one generation (non-wrapping).
    pub fn step(&mut self) {
        let mut next = self.cells.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let cur = self.cells[y * self.w + x];
                next[y * self.w + x] = match cur {
                    1 => 2,
                    2 => 3,
                    3 => {
                        let mut heads = 0;
                        for dy in -1i64..=1 {
                            for dx in -1i64..=1 {
                                if dx == 0 && dy == 0 {
                                    continue;
                                }
                                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                                if nx >= 0
                                    && ny >= 0
                                    && nx < self.w as i64
                                    && ny < self.h as i64
                                    && self.cells[ny as usize * self.w + nx as usize] == 1
                                {
                                    heads += 1;
                                }
                            }
                        }
                        if heads == 1 || heads == 2 { 1 } else { 3 }
                    }
                    _ => 0,
                };
            }
        }
        self.cells = next;
    }

    /// Advances `n` generations.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Number of electron heads on the board.
    #[must_use]
    pub fn count_electrons(&self) -> usize {
        self.cells.iter().filter(|&&c| c == 1).count()
    }
}

/// Life-like automaton on a 3-D grid with the 26-cell Moore
/// neighborhood and a B/S rule (e.g. "B5/S45" for Clouds-like
/// rules).
#[derive(Debug, Clone)]
pub struct LifeLike3D {
    pub w: usize,
    pub h: usize,
    pub d: usize,
    pub cells: Vec<bool>,
    pub birth: [bool; 27],
    pub survive: [bool; 27],
}

impl LifeLike3D {
    /// Parses "B<counts>/S<counts>" where counts are comma-free
    /// digit runs; multi-digit counts (10-26) are written with
    /// parentheses, e.g. "B(10)(11)/S(12)".
    ///
    /// # Errors
    /// `InvalidArgument` on malformed rules.
    pub fn from_rule_string(w: usize, h: usize, d: usize, rule: &str) -> Result<Self, GeomError> {
        assert!(w >= 3 && h >= 3 && d >= 3, "grid must be at least 3x3x3");
        let mut parts = rule.split('/');
        let (b, s) = match (parts.next(), parts.next(), parts.next()) {
            (Some(b), Some(s), None) => (b, s),
            _ => return Err(GeomError::InvalidArgument("rule must be B.../S...")),
        };
        if !b.starts_with(['B', 'b']) || !s.starts_with(['S', 's']) {
            return Err(GeomError::InvalidArgument("rule must be B.../S..."));
        }
        let parse = |spec: &str, table: &mut [bool; 27]| -> Result<(), GeomError> {
            let mut chars = spec.chars().peekable();
            while let Some(ch) = chars.next() {
                match ch {
                    '0'..='9' => table[ch.to_digit(10).unwrap() as usize] = true,
                    '(' => {
                        let mut num = 0usize;
                        for c in chars.by_ref() {
                            if c == ')' {
                                break;
                            }
                            num = num * 10
                                + c.to_digit(10)
                                    .ok_or(GeomError::InvalidArgument("bad count"))?
                                    as usize;
                        }
                        if num > 26 {
                            return Err(GeomError::InvalidArgument("count must be <= 26"));
                        }
                        table[num] = true;
                    }
                    _ => return Err(GeomError::InvalidArgument("unexpected rule character")),
                }
            }
            Ok(())
        };
        let mut birth = [false; 27];
        let mut survive = [false; 27];
        parse(&b[1..], &mut birth)?;
        parse(&s[1..], &mut survive)?;
        Ok(Self { w, h, d, cells: vec![false; w * h * d], birth, survive })
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.h + y) * self.w + x
    }

    /// Advances one generation (toroidal).
    pub fn step(&mut self) {
        let mut next = vec![false; self.cells.len()];
        for z in 0..self.d {
            for y in 0..self.h {
                for x in 0..self.w {
                    let mut n = 0usize;
                    for dz in -1i64..=1 {
                        for dy in -1i64..=1 {
                            for dx in -1i64..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let nx = (x as i64 + dx).rem_euclid(self.w as i64) as usize;
                                let ny = (y as i64 + dy).rem_euclid(self.h as i64) as usize;
                                let nz = (z as i64 + dz).rem_euclid(self.d as i64) as usize;
                                n += usize::from(self.cells[self.index(nx, ny, nz)]);
                            }
                        }
                    }
                    let idx = self.index(x, y, z);
                    next[idx] =
                        if self.cells[idx] { self.survive[n] } else { self.birth[n] };
                }
            }
        }
        self.cells = next;
    }

    /// Number of live cells.
    #[must_use]
    pub fn population(&self) -> usize {
        self.cells.iter().filter(|&&c| c).count()
    }
}

/// SmoothLife parameters (Rafler 2011): inner/outer disc radii and
/// the birth/death sigmoid intervals.
#[derive(Debug, Clone, Copy)]
pub struct SmoothLifeParams {
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub b1: f64,
    pub b2: f64,
    pub d1: f64,
    pub d2: f64,
    /// Sigmoid sharpness for the neighborhood response.
    pub alpha_n: f64,
    /// Sigmoid sharpness for the cell-state mixing.
    pub alpha_m: f64,
}

impl Default for SmoothLifeParams {
    fn default() -> Self {
        Self {
            inner_radius: 3.0,
            outer_radius: 9.0,
            b1: 0.278,
            b2: 0.365,
            d1: 0.267,
            d2: 0.445,
            alpha_n: 0.028,
            alpha_m: 0.147,
        }
    }
}

/// SmoothLife: a continuous-state, continuous-neighborhood
/// generalization of Life, integrated by direct convolution (small
/// grids; no FFT dependency).
#[derive(Debug, Clone)]
pub struct SmoothLife {
    pub w: usize,
    pub h: usize,
    pub field: Vec<f64>,
    pub params: SmoothLifeParams,
}

impl SmoothLife {
    /// Blank field.
    ///
    /// # Panics
    /// Panics unless the grid is at least 8×8.
    #[must_use]
    pub fn new(w: usize, h: usize, params: SmoothLifeParams) -> Self {
        assert!(w >= 8 && h >= 8, "grid must be at least 8x8");
        Self { w, h, field: vec![0.0; w * h], params }
    }

    fn sigmoid(x: f64, a: f64, alpha: f64) -> f64 {
        1.0 / (1.0 + (-(x - a) * 4.0 / alpha).exp())
    }

    /// One smooth timestep of size `dt` (forward Euler on the
    /// transition function).
    pub fn step(&mut self, dt: f64) {
        let p = self.params;
        let ro = p.outer_radius.ceil() as i64;
        let mut next = self.field.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let (mut m, mut mw) = (0.0, 0.0);
                let (mut n, mut nw) = (0.0, 0.0);
                for dy in -ro..=ro {
                    for dx in -ro..=ro {
                        let r = ((dx * dx + dy * dy) as f64).sqrt();
                        if r > p.outer_radius + 0.5 {
                            continue;
                        }
                        let nx = (x as i64 + dx).rem_euclid(self.w as i64) as usize;
                        let ny = (y as i64 + dy).rem_euclid(self.h as i64) as usize;
                        let v = self.field[ny * self.w + nx];
                        // Antialiased disc/annulus membership.
                        let win = (p.inner_radius + 0.5 - r).clamp(0.0, 1.0);
                        let wout = (p.outer_radius + 0.5 - r).clamp(0.0, 1.0) - win;
                        m += v * win;
                        mw += win;
                        n += v * wout;
                        nw += wout;
                    }
                }
                let m = m / mw.max(1.0);
                let n = n / nw.max(1.0);
                // Smooth interval thresholds mixed by the cell state.
                let b = |lo: f64, hi: f64, x: f64, alpha: f64| {
                    Self::sigmoid(x, lo, alpha) * (1.0 - Self::sigmoid(x, hi, alpha))
                };
                let state = Self::sigmoid(m, 0.5, p.alpha_m);
                let lo = p.b1 + (p.d1 - p.b1) * state;
                let hi = p.b2 + (p.d2 - p.b2) * state;
                let target = b(lo, hi, n, p.alpha_n);
                let idx = y * self.w + x;
                next[idx] = (self.field[idx] + dt * (target - self.field[idx])).clamp(0.0, 1.0);
            }
        }
        self.field = next;
    }
}

/// Lenia (Chan 2019): continuous cellular automaton with a smooth
/// ring kernel and a Gaussian growth mapping, integrated by direct
/// convolution.
#[derive(Debug, Clone)]
pub struct Lenia {
    pub w: usize,
    pub h: usize,
    pub field: Vec<f64>,
    /// Kernel radius in cells.
    pub radius: usize,
    /// Precomputed normalized kernel, (2r+1)² row-major.
    pub kernel: Vec<f64>,
    /// Growth-center μ and width σ of the Gaussian growth map.
    pub mu: f64,
    pub sigma: f64,
}

impl Lenia {
    /// Standard Lenia with the smooth ring kernel
    /// exp(4 − 1/(r(1−r))) and growth 2·exp(−(u−μ)²/2σ²) − 1.
    ///
    /// # Panics
    /// Panics unless the grid is at least 2r+1 wide and σ > 0.
    #[must_use]
    pub fn new(w: usize, h: usize, radius: usize, mu: f64, sigma: f64) -> Self {
        assert!(radius >= 2, "kernel radius must be >= 2");
        assert!(w > 2 * radius && h > 2 * radius, "grid too small for the kernel");
        assert!(sigma > 0.0, "sigma must be positive");
        let size = 2 * radius + 1;
        let mut kernel = vec![0.0f64; size * size];
        let mut total = 0.0;
        for dy in 0..size {
            for dx in 0..size {
                let r = (((dx as f64 - radius as f64).powi(2)
                    + (dy as f64 - radius as f64).powi(2))
                .sqrt())
                    / radius as f64;
                if r > 0.0 && r < 1.0 {
                    let v = (4.0 - 1.0 / (r * (1.0 - r))).exp();
                    kernel[dy * size + dx] = v;
                    total += v;
                }
            }
        }
        for k in &mut kernel {
            *k /= total;
        }
        Self { w, h, field: vec![0.0; w * h], radius, kernel, mu, sigma }
    }

    /// One Lenia timestep of size `dt`.
    pub fn step(&mut self, dt: f64) {
        let size = 2 * self.radius + 1;
        let r = self.radius as i64;
        let mut next = self.field.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let mut u = 0.0;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let k = self.kernel[(dy + r) as usize * size + (dx + r) as usize];
                        if k == 0.0 {
                            continue;
                        }
                        let nx = (x as i64 + dx).rem_euclid(self.w as i64) as usize;
                        let ny = (y as i64 + dy).rem_euclid(self.h as i64) as usize;
                        u += k * self.field[ny * self.w + nx];
                    }
                }
                let growth =
                    2.0 * (-(u - self.mu) * (u - self.mu) / (2.0 * self.sigma * self.sigma)).exp()
                        - 1.0;
                let idx = y * self.w + x;
                next[idx] = (self.field[idx] + dt * growth).clamp(0.0, 1.0);
            }
        }
        self.field = next;
    }
}

/// Totalistic rule: the next state is the base-k digit of `code` at
/// the index given by the neighborhood sum.
pub fn totalistic_rule(k: u8, code: u64) -> impl Fn(&[u8]) -> u8 {
    move |neighborhood: &[u8]| {
        let sum: u64 = neighborhood.iter().map(|&c| u64::from(c)).sum();
        let mut c = code;
        for _ in 0..sum {
            c /= u64::from(k);
        }
        (c % u64::from(k)) as u8
    }
}

/// Topples every cell with >= 4 grains (von Neumann neighbors, open
/// boundary: grains fall off the edge) until stable. Returns the
/// number of topplings.
///
/// # Panics
/// Panics unless `grid.len() == w·h`.
pub fn sandpile_abelian(grid: &mut [u32], w: usize, h: usize) -> usize {
    assert_eq!(grid.len(), w * h, "grid size mismatch");
    let mut topplings = 0usize;
    let mut queue: Vec<usize> = (0..grid.len()).filter(|&i| grid[i] >= 4).collect();
    while let Some(idx) = queue.pop() {
        while grid[idx] >= 4 {
            grid[idx] -= 4;
            topplings += 1;
            let (x, y) = (idx % w, idx / w);
            for (nx, ny) in
                [(x as i64 - 1, y as i64), (x as i64 + 1, y as i64), (x as i64, y as i64 - 1), (x as i64, y as i64 + 1)]
            {
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                    let n = ny as usize * w + nx as usize;
                    grid[n] += 1;
                    if grid[n] == 4 {
                        queue.push(n);
                    }
                }
            }
        }
    }
    topplings
}

/// Identity element of the abelian sandpile group on the w×h grid:
/// stabilize(2·δ − stabilize(2·δ)) with δ the all-6 configuration.
/// Adding it to any recurrent configuration and stabilizing returns
/// that configuration.
#[must_use]
pub fn sandpile_identity(w: usize, h: usize) -> Vec<u32> {
    let mut a = vec![6u32; w * h];
    sandpile_abelian(&mut a, w, h);
    let mut b: Vec<u32> = a.iter().map(|&v| 6 - v).collect();
    sandpile_abelian(&mut b, w, h);
    b
}

/// Drossel-Schwabl forest fire: 0 empty, 1 tree, 2 burning. Burning
/// cells become empty; trees with a burning neighbor (or struck by
/// lightning with probability `p_lightning`) burn; empty cells grow
/// a tree with probability `p_grow`. Returns every generation.
///
/// # Panics
/// Panics unless the grid is at least 3×3.
#[must_use]
pub fn forest_fire(
    w: usize,
    h: usize,
    p_grow: f64,
    p_lightning: f64,
    steps: usize,
    rng: &mut Rng,
) -> Vec<Vec<u8>> {
    assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
    let mut cells: Vec<u8> = (0..w * h).map(|_| u8::from(rng.next_f64() < 0.5)).collect();
    let mut out = Vec::with_capacity(steps + 1);
    out.push(cells.clone());
    for _ in 0..steps {
        let mut next = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                next[idx] = match cells[idx] {
                    2 => 0,
                    1 => {
                        let mut burning = false;
                        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                            let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                            let ny = (y as i64 + dy).rem_euclid(h as i64) as usize;
                            burning |= cells[ny * w + nx] == 2;
                        }
                        if burning || rng.next_f64() < p_lightning { 2 } else { 1 }
                    }
                    _ => u8::from(rng.next_f64() < p_grow),
                };
            }
        }
        cells = next;
        out.push(cells.clone());
    }
    out
}

/// Greenberg-Hastings excitable medium: state 0 rests, 1 fires,
/// 2..states-1 are refractory. A resting cell fires when a von
/// Neumann neighbor fires; every other state advances and wraps to
/// rest. Returns every generation from a random start.
///
/// # Panics
/// Panics unless the grid is at least 3×3 and `states >= 3`.
#[must_use]
pub fn greenberg_hastings(
    w: usize,
    h: usize,
    states: u8,
    steps: usize,
    rng: &mut Rng,
) -> Vec<Vec<u8>> {
    assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
    assert!(states >= 3, "need at least 3 states");
    let mut cells: Vec<u8> =
        (0..w * h).map(|_| (rng.next_f64() * f64::from(states)) as u8 % states).collect();
    let mut out = Vec::with_capacity(steps + 1);
    out.push(cells.clone());
    for _ in 0..steps {
        let mut next = cells.clone();
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if cells[idx] == 0 {
                    let mut excited = false;
                    for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                        let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                        let ny = (y as i64 + dy).rem_euclid(h as i64) as usize;
                        excited |= cells[ny * w + nx] == 1;
                    }
                    if excited {
                        next[idx] = 1;
                    }
                } else {
                    next[idx] = (cells[idx] + 1) % states;
                }
            }
        }
        cells = next;
        out.push(cells.clone());
    }
    out
}

/// Synchronous majority rule: each cell adopts the majority state of
/// its Moore neighborhood (including itself; ties keep the state).
///
/// # Panics
/// Panics unless `cells.len() == w·h`.
pub fn majority_rule(cells: &mut [bool], w: usize, h: usize, steps: usize) {
    assert_eq!(cells.len(), w * h, "grid size mismatch");
    for _ in 0..steps {
        let snapshot = cells.to_vec();
        for y in 0..h {
            for x in 0..w {
                let mut alive = 0i32;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                        let ny = (y as i64 + dy).rem_euclid(h as i64) as usize;
                        alive += i32::from(snapshot[ny * w + nx]);
                    }
                }
                if alive > 5 {
                    cells[y * w + x] = true;
                } else if alive < 4 {
                    cells[y * w + x] = false;
                }
            }
        }
    }
}

/// Voter model: each step a random cell copies a random von Neumann
/// neighbor (`steps` single-cell updates).
///
/// # Panics
/// Panics unless `cells.len() == w·h`.
pub fn voter_model(cells: &mut [bool], w: usize, h: usize, steps: usize, rng: &mut Rng) {
    assert_eq!(cells.len(), w * h, "grid size mismatch");
    for _ in 0..steps {
        let idx = (rng.next_f64() * (w * h) as f64) as usize % (w * h);
        let (x, y) = (idx % w, idx / w);
        let dirs = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)];
        let (dx, dy) = dirs[(rng.next_f64() * 4.0) as usize % 4];
        let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
        let ny = (y as i64 + dy).rem_euclid(h as i64) as usize;
        cells[idx] = cells[ny * w + nx];
    }
}

/// Schelling segregation: two agent types (1, 2) plus vacancies (0).
/// Unhappy agents (fewer than `threshold` same-type fraction among
/// occupied Moore neighbors) move to random vacancies. Returns the
/// final segregation index: the mean same-type fraction over
/// occupied neighbors of all agents (0.5 = mixed, 1 = segregated).
///
/// # Panics
/// Panics unless `grid.len() == w·h` and threshold is in [0, 1].
pub fn schelling_segregation(
    grid: &mut [u8],
    w: usize,
    h: usize,
    threshold: f64,
    steps: usize,
    rng: &mut Rng,
) -> f64 {
    assert_eq!(grid.len(), w * h, "grid size mismatch");
    assert!((0.0..=1.0).contains(&threshold), "threshold must be in [0, 1]");
    let same_fraction = |grid: &[u8], x: usize, y: usize| -> Option<f64> {
        let me = grid[y * w + x];
        let mut same = 0usize;
        let mut occupied = 0usize;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = (x as i64 + dx).rem_euclid(w as i64) as usize;
                let ny = (y as i64 + dy).rem_euclid(h as i64) as usize;
                let v = grid[ny * w + nx];
                if v != 0 {
                    occupied += 1;
                    same += usize::from(v == me);
                }
            }
        }
        (occupied > 0).then(|| same as f64 / occupied as f64)
    };
    for _ in 0..steps {
        // Collect unhappy agents and vacancies.
        let mut unhappy = Vec::new();
        let mut vacant = Vec::new();
        for y in 0..h {
            for x in 0..w {
                match grid[y * w + x] {
                    0 => vacant.push(y * w + x),
                    _ => {
                        if same_fraction(grid, x, y).is_some_and(|f| f < threshold) {
                            unhappy.push(y * w + x);
                        }
                    }
                }
            }
        }
        if unhappy.is_empty() || vacant.is_empty() {
            break;
        }
        for &agent in &unhappy {
            if vacant.is_empty() {
                break;
            }
            let vi = (rng.next_f64() * vacant.len() as f64) as usize % vacant.len();
            let target = vacant.swap_remove(vi);
            grid[target] = grid[agent];
            grid[agent] = 0;
            vacant.push(agent);
        }
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 0..h {
        for x in 0..w {
            if grid[y * w + x] != 0 {
                if let Some(f) = same_fraction(grid, x, y) {
                    sum += f;
                    count += 1;
                }
            }
        }
    }
    if count > 0 { sum / count as f64 } else { 0.5 }
}

fn laplacian(field: &[f64], w: usize, h: usize, x: usize, y: usize) -> f64 {
    let idx = |x: i64, y: i64| -> f64 {
        field[(y.rem_euclid(h as i64) as usize) * w + x.rem_euclid(w as i64) as usize]
    };
    let (xi, yi) = (x as i64, y as i64);
    idx(xi - 1, yi) + idx(xi + 1, yi) + idx(xi, yi - 1) + idx(xi, yi + 1) - 4.0 * idx(xi, yi)
}

/// Gray-Scott reaction-diffusion: ∂u = Dᵤ∇²u − uv² + F(1−u),
/// ∂v = Dᵥ∇²v + uv² − (F+k)v (Pearson 1993).
#[derive(Debug, Clone)]
pub struct GrayScott {
    pub w: usize,
    pub h: usize,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub du: f64,
    pub dv: f64,
    pub feed: f64,
    pub kill: f64,
    pub dt: f64,
}

impl GrayScott {
    /// Uniform u = 1, v = 0 state with standard diffusion rates.
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3.
    #[must_use]
    pub fn new(w: usize, h: usize, feed: f64, kill: f64) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        Self {
            w,
            h,
            u: vec![1.0; w * h],
            v: vec![0.0; w * h],
            du: 0.16,
            dv: 0.08,
            feed,
            kill,
            dt: 1.0,
        }
    }

    /// Pearson's mitosis regime (F = 0.0367, k = 0.0649).
    #[must_use]
    pub fn mitosis(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.0367, 0.0649)
    }

    /// Coral growth regime (F = 0.0545, k = 0.062).
    #[must_use]
    pub fn coral(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.0545, 0.062)
    }

    /// Spots (F = 0.03, k = 0.062).
    #[must_use]
    pub fn spots(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.03, 0.062)
    }

    /// Worms (F = 0.046, k = 0.063).
    #[must_use]
    pub fn worms(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.046, 0.063)
    }

    /// Maze-like labyrinths (F = 0.029, k = 0.057).
    #[must_use]
    pub fn maze(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.029, 0.057)
    }

    /// Holes (F = 0.039, k = 0.058).
    #[must_use]
    pub fn holes(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.039, 0.058)
    }

    /// Travelling waves (F = 0.014, k = 0.045).
    #[must_use]
    pub fn waves(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.014, 0.045)
    }

    /// Solitons (F = 0.03, k = 0.06).
    #[must_use]
    pub fn solitons(w: usize, h: usize) -> Self {
        Self::new(w, h, 0.03, 0.06)
    }

    /// Seeds a square of v = 1, u = 0.5 (the usual perturbation).
    pub fn seed_square(&mut self, x: usize, y: usize, size: usize) {
        for j in 0..size {
            for i in 0..size {
                let idx = ((y + j) % self.h) * self.w + (x + i) % self.w;
                self.u[idx] = 0.5;
                self.v[idx] = 1.0;
            }
        }
    }

    /// One forward-Euler step.
    pub fn step(&mut self) {
        let mut un = self.u.clone();
        let mut vn = self.v.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let (u, v) = (self.u[idx], self.v[idx]);
                let uvv = u * v * v;
                un[idx] = (u
                    + self.dt
                        * (self.du * laplacian(&self.u, self.w, self.h, x, y) - uvv
                            + self.feed * (1.0 - u)))
                    .clamp(0.0, 1.5);
                vn[idx] = (v
                    + self.dt
                        * (self.dv * laplacian(&self.v, self.w, self.h, x, y) + uvv
                            - (self.feed + self.kill) * v))
                    .clamp(0.0, 1.5);
            }
        }
        self.u = un;
        self.v = vn;
    }

    /// Runs `n` steps.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

/// Gierer-Meinhardt activator-inhibitor system:
/// ∂a = Dₐ∇²a + a²/h − μa + ρ, ∂h = Dₕ∇²h + a² − νh.
#[derive(Debug, Clone)]
pub struct Turing {
    pub w: usize,
    pub h: usize,
    pub activator: Vec<f64>,
    pub inhibitor: Vec<f64>,
    pub da: f64,
    pub dh: f64,
    pub mu: f64,
    pub nu: f64,
    pub rho: f64,
    pub dt: f64,
}

impl Turing {
    /// Near-homogeneous start with small random perturbations.
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3.
    #[must_use]
    pub fn new(w: usize, h: usize, rng: &mut Rng) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        let activator = (0..w * h).map(|_| 1.0 + 0.01 * (rng.next_f64() - 0.5)).collect();
        let inhibitor = (0..w * h).map(|_| 1.0 + 0.01 * (rng.next_f64() - 0.5)).collect();
        Self { w, h, activator, inhibitor, da: 0.02, dh: 0.5, mu: 1.0, nu: 1.2, rho: 0.05, dt: 0.05 }
    }

    /// One forward-Euler step.
    pub fn step(&mut self) {
        let mut an = self.activator.clone();
        let mut hn = self.inhibitor.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let (a, hh) = (self.activator[idx], self.inhibitor[idx]);
                an[idx] = (a
                    + self.dt
                        * (self.da * laplacian(&self.activator, self.w, self.h, x, y)
                            + a * a / hh.max(1e-9)
                            - self.mu * a
                            + self.rho))
                    .max(0.0);
                hn[idx] = (hh
                    + self.dt
                        * (self.dh * laplacian(&self.inhibitor, self.w, self.h, x, y) + a * a
                            - self.nu * hh))
                    .max(1e-9);
            }
        }
        self.activator = an;
        self.inhibitor = hn;
    }
}

/// FitzHugh-Nagumo excitable medium: ∂v = D∇²v + v − v³/3 − w,
/// ∂w = ε(v + a − b·w), with no-flux boundaries (wrapped copies
/// annihilate spirals).
#[derive(Debug, Clone)]
pub struct FitzHughNagumo {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f64>,
    pub w_: Vec<f64>,
    pub a: f64,
    pub b: f64,
    pub eps: f64,
    pub d: f64,
    pub dt: f64,
}

impl FitzHughNagumo {
    /// Uniform resting state, parameterized in the excitable regime
    /// that supports rotating spirals on modest grids
    /// (a = 0.5, b = 0.8, ε = 0.05, D = 0.3).
    ///
    /// # Panics
    /// Panics unless the grid is at least 8×8.
    #[must_use]
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w >= 8 && h >= 8, "grid must be at least 8x8");
        Self {
            w,
            h,
            v: vec![-1.2; w * h],
            w_: vec![-0.6; w * h],
            a: 0.5,
            b: 0.8,
            eps: 0.05,
            d: 0.3,
            dt: 0.1,
        }
    }

    /// Seeds a phase-distributed spiral: v and w wind once around
    /// the grid center, which relaxes into a rotating spiral wave.
    pub fn spiral_wave_seed(&mut self) {
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let phi = (y as f64 - self.h as f64 / 2.0)
                    .atan2(x as f64 - self.w as f64 / 2.0);
                self.v[idx] = 2.0 * phi.cos();
                self.w_[idx] = phi.sin();
            }
        }
    }

    /// One forward-Euler step (no-flux boundaries).
    pub fn step(&mut self) {
        let mut vn = self.v.clone();
        let mut wn = self.w_.clone();
        let cl = |i: i64, n: usize| -> usize { i.clamp(0, n as i64 - 1) as usize };
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let (v, w) = (self.v[idx], self.w_[idx]);
                let (xi, yi) = (x as i64, y as i64);
                let lap = self.v[cl(yi - 1, self.h) * self.w + x]
                    + self.v[cl(yi + 1, self.h) * self.w + x]
                    + self.v[y * self.w + cl(xi - 1, self.w)]
                    + self.v[y * self.w + cl(xi + 1, self.w)]
                    - 4.0 * v;
                vn[idx] = v + self.dt * (self.d * lap + v - v * v * v / 3.0 - w);
                wn[idx] = w + self.dt * self.eps * (v + self.a - self.b * w);
            }
        }
        self.v = vn;
        self.w_ = wn;
    }

    /// Runs `n` steps.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

/// Two-variable Oregonator model of the Belousov-Zhabotinsky
/// reaction: ∂u = ∇²u + (u(1−u) − f·v(u−q)/(u+q))/ε, ∂v = ∇²v·Dᵥ + u − v.
#[derive(Debug, Clone)]
pub struct BelousovZhabotinsky {
    pub w: usize,
    pub h: usize,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub eps: f64,
    pub f: f64,
    pub q: f64,
    pub dv: f64,
    pub dt: f64,
}

impl BelousovZhabotinsky {
    /// Resting medium with an excited spot in the center.
    ///
    /// # Panics
    /// Panics unless the grid is at least 8×8.
    #[must_use]
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w >= 8 && h >= 8, "grid must be at least 8x8");
        let mut u = vec![0.01; w * h];
        let v = vec![0.01; w * h];
        for j in 0..3 {
            for i in 0..3 {
                u[(h / 2 + j) * w + w / 2 + i] = 0.8;
            }
        }
        Self { w, h, u, v, eps: 0.02, f: 1.4, q: 0.002, dv: 0.6, dt: 0.001 }
    }

    /// One forward-Euler step.
    pub fn step(&mut self) {
        let mut un = self.u.clone();
        let mut vn = self.v.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let (u, v) = (self.u[idx], self.v[idx]);
                let reaction = (u * (1.0 - u) - self.f * v * (u - self.q) / (u + self.q)) / self.eps;
                un[idx] =
                    (u + self.dt * (laplacian(&self.u, self.w, self.h, x, y) + reaction)).max(0.0);
                vn[idx] = (v
                    + self.dt * (self.dv * laplacian(&self.v, self.w, self.h, x, y) + u - v))
                    .max(0.0);
            }
        }
        self.u = un;
        self.v = vn;
    }
}

/// Brusselator: ∂u = Dᵤ∇²u + A − (B+1)u + u²v, ∂v = Dᵥ∇²v + Bu − u²v.
#[derive(Debug, Clone)]
pub struct Brusselator {
    pub w: usize,
    pub h: usize,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub a: f64,
    pub b: f64,
    pub du: f64,
    pub dv: f64,
    pub dt: f64,
}

impl Brusselator {
    /// Starts at the homogeneous fixed point (A, B/A) with small
    /// random perturbations; B > 1 + A² puts it in the Turing/
    /// oscillatory regime.
    ///
    /// # Panics
    /// Panics unless the grid is at least 3×3 and `a > 0`.
    #[must_use]
    pub fn new(w: usize, h: usize, a: f64, b: f64, rng: &mut Rng) -> Self {
        assert!(w >= 3 && h >= 3, "grid must be at least 3x3");
        assert!(a > 0.0, "A must be positive");
        let u = (0..w * h).map(|_| a + 0.01 * (rng.next_f64() - 0.5)).collect();
        let v = (0..w * h).map(|_| b / a + 0.01 * (rng.next_f64() - 0.5)).collect();
        Self { w, h, u, v, a, b, du: 2.0, dv: 16.0, dt: 0.005 }
    }

    /// One forward-Euler step.
    pub fn step(&mut self) {
        let mut un = self.u.clone();
        let mut vn = self.v.clone();
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = y * self.w + x;
                let (u, v) = (self.u[idx], self.v[idx]);
                un[idx] = (u
                    + self.dt
                        * (self.du * laplacian(&self.u, self.w, self.h, x, y) + self.a
                            - (self.b + 1.0) * u
                            + u * u * v))
                    .max(0.0);
                vn[idx] = (v
                    + self.dt
                        * (self.dv * laplacian(&self.v, self.w, self.h, x, y) + self.b * u
                            - u * u * v))
                    .max(0.0);
            }
        }
        self.u = un;
        self.v = vn;
    }
}

/// Generic 1-D two-species reaction-diffusion by forward Euler with
/// zero-flux boundaries: `f(u, v)` returns the two reaction rates.
///
/// # Panics
/// Panics unless the arrays match and have at least 3 cells, and
/// `dx > 0`, `dt > 0`.
#[allow(clippy::too_many_arguments)]
pub fn reaction_diffusion_1d(
    u: &mut [f64],
    v: &mut [f64],
    f: &dyn Fn(f64, f64) -> (f64, f64),
    du: f64,
    dv: f64,
    dt: f64,
    dx: f64,
    steps: usize,
) {
    assert_eq!(u.len(), v.len(), "field size mismatch");
    assert!(u.len() >= 3, "need at least 3 cells");
    assert!(dx > 0.0 && dt > 0.0, "dx and dt must be positive");
    let n = u.len();
    let inv = 1.0 / (dx * dx);
    for _ in 0..steps {
        let us = u.to_vec();
        let vs = v.to_vec();
        for i in 0..n {
            let (im, ip) = (i.saturating_sub(1), (i + 1).min(n - 1));
            let lap_u = (us[im] - 2.0 * us[i] + us[ip]) * inv;
            let lap_v = (vs[im] - 2.0 * vs[i] + vs[ip]) * inv;
            let (fu, fv) = f(us[i], vs[i]);
            u[i] = us[i] + dt * (du * lap_u + fu);
            v[i] = vs[i] + dt * (dv * lap_v + fv);
        }
    }
}

/// Diffusion-limited aggregation on a lattice: random walkers
/// launched from a circle stick to the growing cluster with the
/// given probability. Returns the cluster mask (row-major).
///
/// # Panics
/// Panics unless the grid is at least 16×16 and stickiness is in
/// (0, 1].
#[must_use]
pub fn diffusion_limited_aggregation(
    w: usize,
    h: usize,
    particles: usize,
    stickiness: f64,
    rng: &mut Rng,
) -> Vec<bool> {
    assert!(w >= 16 && h >= 16, "grid must be at least 16x16");
    assert!(stickiness > 0.0 && stickiness <= 1.0, "stickiness in (0, 1]");
    let (cx, cy) = (w as i64 / 2, h as i64 / 2);
    let mut cluster = vec![false; w * h];
    cluster[cy as usize * w + cx as usize] = true;
    let mut radius = 2.0f64;
    let max_radius = (w.min(h) as f64) / 2.0 - 2.0;
    for _ in 0..particles {
        if radius >= max_radius {
            break;
        }
        // Launch on a circle just outside the cluster.
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let launch = radius + 2.0;
        let mut x = cx + (launch * angle.cos()).round() as i64;
        let mut y = cy + (launch * angle.sin()).round() as i64;
        let kill = (launch * 2.0 + 4.0).min(max_radius + 4.0);
        loop {
            let dir = (rng.next_f64() * 4.0) as u64 % 4;
            match dir {
                0 => x += 1,
                1 => x -= 1,
                2 => y += 1,
                _ => y -= 1,
            }
            let dx = (x - cx) as f64;
            let dy = (y - cy) as f64;
            if (dx * dx + dy * dy).sqrt() > kill {
                // Wandered too far: relaunch.
                let angle = rng.next_f64() * std::f64::consts::TAU;
                x = cx + (launch * angle.cos()).round() as i64;
                y = cy + (launch * angle.sin()).round() as i64;
                continue;
            }
            if x < 1 || y < 1 || x >= w as i64 - 1 || y >= h as i64 - 1 {
                continue;
            }
            let touching = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                .iter()
                .any(|&(nx, ny)| cluster[ny as usize * w + nx as usize]);
            if touching && rng.next_f64() < stickiness {
                cluster[y as usize * w + x as usize] = true;
                let r = ((x - cx).pow(2) + (y - cy).pow(2)) as f64;
                radius = radius.max(r.sqrt());
                break;
            }
        }
    }
    cluster
}

/// Eden growth: repeatedly turns a random perimeter cell of the
/// cluster on (compact growth with a rough boundary).
///
/// # Panics
/// Panics unless the grid is at least 8×8.
#[must_use]
pub fn eden_growth(w: usize, h: usize, steps: usize, rng: &mut Rng) -> Vec<bool> {
    assert!(w >= 8 && h >= 8, "grid must be at least 8x8");
    let mut cluster = vec![false; w * h];
    let start = (h / 2) * w + w / 2;
    cluster[start] = true;
    let mut perimeter: Vec<usize> = Vec::new();
    let neighbors = |idx: usize| -> Vec<usize> {
        let (x, y) = ((idx % w) as i64, (idx / w) as i64);
        [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            .iter()
            .filter(|&&(nx, ny)| nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64)
            .map(|&(nx, ny)| ny as usize * w + nx as usize)
            .collect()
    };
    for n in neighbors(start) {
        perimeter.push(n);
    }
    for _ in 0..steps {
        if perimeter.is_empty() {
            break;
        }
        let k = (rng.next_f64() * perimeter.len() as f64) as usize % perimeter.len();
        let cell = perimeter.swap_remove(k);
        if cluster[cell] {
            continue;
        }
        cluster[cell] = true;
        for n in neighbors(cell) {
            if !cluster[n] {
                perimeter.push(n);
            }
        }
    }
    cluster
}

/// Invasion percolation: cells get random strengths; growth always
/// invades the weakest perimeter cell, until the cluster touches a
/// boundary. Returns the invaded mask.
///
/// # Panics
/// Panics unless the grid is at least 8×8.
#[must_use]
pub fn invasion_percolation(w: usize, h: usize, rng: &mut Rng) -> Vec<bool> {
    assert!(w >= 8 && h >= 8, "grid must be at least 8x8");
    let strengths: Vec<f64> = (0..w * h).map(|_| rng.next_f64()).collect();
    let mut invaded = vec![false; w * h];
    let start = (h / 2) * w + w / 2;
    invaded[start] = true;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
    let push_neighbors = |heap: &mut BinaryHeap<(Reverse<u64>, usize)>, idx: usize| {
        let (x, y) = ((idx % w) as i64, (idx / w) as i64);
        for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                let n = ny as usize * w + nx as usize;
                heap.push((Reverse(strengths[n].to_bits()), n));
            }
        }
    };
    push_neighbors(&mut heap, start);
    while let Some((_, idx)) = heap.pop() {
        if invaded[idx] {
            continue;
        }
        invaded[idx] = true;
        let (x, y) = (idx % w, idx / w);
        if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
            break;
        }
        push_neighbors(&mut heap, idx);
    }
    invaded
}

struct Dsu {
    parent: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect() }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Labels the 4-connected clusters of `grid` (labels start at 1;
/// 0 = off) and reports whether any cluster spans top to bottom.
///
/// # Panics
/// Panics unless `grid.len() == w·h`.
#[must_use]
pub fn percolation_cluster(grid: &[bool], w: usize, h: usize) -> (Vec<u32>, bool) {
    assert_eq!(grid.len(), w * h, "grid size mismatch");
    let mut dsu = Dsu::new(w * h);
    for y in 0..h {
        for x in 0..w {
            if !grid[y * w + x] {
                continue;
            }
            if x + 1 < w && grid[y * w + x + 1] {
                dsu.union(y * w + x, y * w + x + 1);
            }
            if y + 1 < h && grid[(y + 1) * w + x] {
                dsu.union(y * w + x, (y + 1) * w + x);
            }
        }
    }
    let mut labels = vec![0u32; w * h];
    let mut next = 1u32;
    let mut label_of = std::collections::HashMap::new();
    for i in 0..w * h {
        if grid[i] {
            let root = dsu.find(i);
            let label = *label_of.entry(root).or_insert_with(|| {
                let l = next;
                next += 1;
                l
            });
            labels[i] = label;
        }
    }
    let top: std::collections::HashSet<u32> =
        (0..w).filter(|&x| grid[x]).map(|x| labels[x]).collect();
    let spans = (0..w)
        .filter(|&x| grid[(h - 1) * w + x])
        .any(|x| top.contains(&labels[(h - 1) * w + x]));
    (labels, spans)
}

/// Site-percolation threshold estimate: cells are enabled in a
/// random order until a cluster spans top to bottom; the spanning
/// fraction, averaged over `trials`, estimates p_c ≈ 0.5927 on the
/// square lattice.
///
/// # Panics
/// Panics unless the grid is at least 8×8 and `trials >= 1`.
#[must_use]
pub fn percolation_threshold_estimate(w: usize, h: usize, trials: usize, rng: &mut Rng) -> f64 {
    assert!(w >= 8 && h >= 8, "grid must be at least 8x8");
    assert!(trials >= 1, "need at least one trial");
    let mut total = 0.0;
    // Virtual top/bottom nodes for fast spanning checks.
    let top = w * h;
    let bottom = w * h + 1;
    for _ in 0..trials {
        let mut order: Vec<usize> = (0..w * h).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize % (i + 1);
            order.swap(i, j);
        }
        let mut open = vec![false; w * h];
        let mut dsu = Dsu::new(w * h + 2);
        let mut added = 0usize;
        for &idx in &order {
            open[idx] = true;
            added += 1;
            let (x, y) = (idx % w, idx / w);
            if y == 0 {
                dsu.union(idx, top);
            }
            if y == h - 1 {
                dsu.union(idx, bottom);
            }
            let (xi, yi) = (x as i64, y as i64);
            for (nx, ny) in [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)] {
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                    let n = ny as usize * w + nx as usize;
                    if open[n] {
                        dsu.union(idx, n);
                    }
                }
            }
            if dsu.find(top) == dsu.find(bottom) {
                break;
            }
        }
        total += added as f64 / (w * h) as f64;
    }
    total / trials as f64
}

/// Mass-radius fractal dimension of a cluster mask: the slope of
/// ln N(r) versus ln r, where N(r) counts cluster cells within
/// distance r of the cluster centroid (radii doubling from 3 up to
/// 70% of the cluster extent, which avoids finite-size edge bias
/// that plagues box counting on sparse clusters).
///
/// # Panics
/// Panics unless `cluster.len() == w·h` and the cluster has at
/// least 10 cells.
#[must_use]
pub fn dla_fractal_dimension(cluster: &[bool], w: usize, h: usize) -> f64 {
    assert_eq!(cluster.len(), w * h, "grid size mismatch");
    let cells: Vec<(f64, f64)> = (0..w * h)
        .filter(|&i| cluster[i])
        .map(|i| ((i % w) as f64, (i / w) as f64))
        .collect();
    assert!(cells.len() >= 10, "cluster too small");
    let cx = cells.iter().map(|c| c.0).sum::<f64>() / cells.len() as f64;
    let cy = cells.iter().map(|c| c.1).sum::<f64>() / cells.len() as f64;
    let dists: Vec<f64> = cells
        .iter()
        .map(|&(x, y)| ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt())
        .collect();
    let r_max = dists.iter().cloned().fold(0.0f64, f64::max) * 0.7;
    let mut fit = Vec::new();
    let mut r = 3.0f64;
    while r <= r_max {
        let n = dists.iter().filter(|&&d| d <= r).count();
        if n > 0 {
            fit.push((r.ln(), (n as f64).ln()));
        }
        r *= 1.5;
    }
    assert!(fit.len() >= 2, "cluster too small for a mass-radius fit");
    let n = fit.len() as f64;
    let sx: f64 = fit.iter().map(|p| p.0).sum();
    let sy: f64 = fit.iter().map(|p| p.1).sum();
    let sxx: f64 = fit.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = fit.iter().map(|p| p.0 * p.1).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_90_is_pascal_mod_2() {
        let mut ca = Ca1D::new(90, 129, false);
        ca.seed_center();
        let rows = ca.run(32);
        // Row n = binomial coefficients C(n, k) mod 2.
        for (n, row) in rows.iter().enumerate().take(33) {
            for (i, &alive) in row.iter().enumerate() {
                let k = i as i64 - (64 - n as i64);
                let expected = if k < 0 || k > 2 * n as i64 || k % 2 != 0 {
                    false
                } else {
                    // C(n, k/2) mod 2 via Lucas: k/2 AND-submask of n.
                    let kk = (k / 2) as u64;
                    kk & n as u64 == kk
                };
                assert_eq!(alive, expected, "row {n} cell {i}");
            }
        }
        assert!(rule_is_additive(90));
        assert!(rule_is_additive(150));
        assert!(!rule_is_additive(110));
        assert!(!rule_is_additive(30));
        assert!(!rule_table(90)[0b101]);
        assert!(rule_table(90)[0b100]);
        // Classification heuristics on the textbook examples.
        assert_eq!(rule_classify_wolfram(0), 1, "rule 0 dies");
        assert_eq!(rule_classify_wolfram(4), 2, "rule 4 freezes");
        assert_eq!(rule_classify_wolfram(30), 3, "rule 30 is chaotic");
    }

    #[test]
    fn test_glider_and_still_lifes() {
        let mut life = LifeLike::conway(20, 20);
        life.place(5, 5, &patterns::glider());
        let before = life.cells.clone();
        life.run(4);
        // Glider translated by (1, 1): compare against a fresh
        // placement one cell diagonally.
        let mut expect = LifeLike::conway(20, 20);
        expect.place(6, 6, &patterns::glider());
        assert_eq!(life.cells, expect.cells, "glider moved (1, 1) in 4 steps");
        assert_ne!(before, life.cells);
        // Block is a still life; beehive too.
        let mut block = LifeLike::conway(10, 10);
        block.place(4, 4, &patterns::block());
        assert!(block.is_still_life());
        assert_eq!(block.detect_period(5), Some(1));
        let mut hive = LifeLike::conway(10, 10);
        hive.place(3, 3, &patterns::beehive());
        assert!(hive.is_still_life());
        // Pulsar has period 3, blinker 2, pentadecathlon 15.
        let mut pulsar = LifeLike::conway(21, 21);
        pulsar.place(4, 4, &patterns::pulsar());
        assert_eq!(pulsar.detect_period(10), Some(3));
        let mut blinker = LifeLike::conway(9, 9);
        blinker.place(3, 4, &patterns::blinker());
        assert_eq!(blinker.detect_period(4), Some(2));
        let mut penta = LifeLike::conway(18, 13);
        penta.place(4, 5, &patterns::pentadecathlon());
        assert_eq!(penta.detect_period(20), Some(15));
    }

    #[test]
    fn test_gosper_gun_emits_gliders() {
        let mut life = LifeLike::conway(80, 60);
        life.wrap = false;
        life.place(2, 2, &patterns::gosper_gun());
        life.run(30);
        let p30 = life.population();
        life.run(30);
        let p60 = life.population();
        life.run(30);
        let p90 = life.population();
        assert_eq!(p60 - p30, 5, "one glider (5 cells) per 30 generations");
        assert_eq!(p90 - p60, 5);
    }

    #[test]
    fn test_rle_rule_strings_and_tostring() {
        // Glider in RLE.
        let mut a = LifeLike::conway(12, 12);
        a.place_rle(3, 3, "bob$2bo$3o!").expect("valid RLE");
        let mut b = LifeLike::conway(12, 12);
        b.place(3, 3, &patterns::glider());
        assert_eq!(a.cells, b.cells, "RLE glider matches");
        assert!(a.place_rle(0, 0, "2x!").is_err());
        // HighLife replicator rule parses; Seeds has empty survival.
        let hl = LifeLike::from_rule_string(10, 10, "B36/S23").expect("HighLife");
        assert!(hl.birth[3] && hl.birth[6] && hl.survive[2] && !hl.survive[1]);
        let seeds = LifeLike::from_rule_string(10, 10, "B2/S").expect("Seeds");
        assert!(seeds.birth[2] && seeds.survive.iter().all(|&s| !s));
        assert!(LifeLike::from_rule_string(10, 10, "3/23").is_err());
        // to_string round-trips the block.
        let mut blk = LifeLike::conway(4, 4);
        blk.place(1, 1, &patterns::block());
        assert_eq!(blk.to_string(), "....\n.OO.\n.OO.\n....\n");
        let bb = blk.bounding_box().expect("live cells");
        assert_eq!((bb.min.x, bb.min.y, bb.max.x, bb.max.y), (1.0, 1.0, 2.0, 2.0));
    }

    #[test]
    fn test_langton_ant_and_turmite() {
        let mut ant = LangtonsAnt::new(400, 400, "RL");
        // The classic ant builds a highway after ~10000 steps.
        ant.run(11_000);
        assert!(ant.highway_detected(), "RL ant builds its highway");
        // Before the highway there is no such periodicity.
        let mut young = LangtonsAnt::new(400, 400, "RL");
        young.run(500);
        assert!(!young.highway_detected());
        // A turmite that mimics the classic ant behaves identically.
        let mut t = Turmite::new(64, 64, vec![vec![(1, 1, 0), (0, 3, 0)]]);
        let mut a = LangtonsAnt::new(64, 64, "RL");
        for _ in 0..500 {
            t.step();
            a.step();
        }
        assert_eq!(t.pos, a.pos, "turmite table reproduces the ant");
    }

    #[test]
    fn test_brain_wireworld_cyclic_life3d() {
        // Brian's Brain: a pair of firing cells dies out into waves;
        // states remain in {0, 1, 2}.
        let mut brain = BriansBrain::new(16, 16);
        brain.cells[8 * 16 + 8] = 2;
        brain.cells[8 * 16 + 9] = 2;
        for _ in 0..10 {
            brain.step();
            assert!(brain.cells.iter().all(|&c| c <= 2));
        }
        // Wireworld: an electron travels down a straight wire and
        // falls off the end.
        let mut ww = Wireworld::from_string("TH######").expect("circuit");
        assert_eq!(ww.count_electrons(), 1);
        let mut alive_steps = 0;
        for _ in 0..12 {
            ww.step();
            if ww.count_electrons() == 1 {
                alive_steps += 1;
            }
        }
        assert!(alive_steps >= 4, "electron traveled the wire ({alive_steps})");
        assert_eq!(ww.count_electrons(), 0, "electron left the open end");
        // Cyclic CA settles into rotating states.
        let mut rng = Rng::new(5);
        let mut cca = CyclicCa::new(24, 24, 8, 1, 1, &mut rng);
        cca.run(30);
        assert!(cca.cells.iter().all(|&c| c < 8));
        // 3-D life: a small blob under B5/S45 stays bounded.
        let mut l3 = LifeLike3D::from_rule_string(12, 12, 12, "B5/S45").expect("rule");
        for z in 5..7 {
            for y in 5..7 {
                for x in 5..7 {
                    let idx = (z * 12 + y) * 12 + x;
                    l3.cells[idx] = true;
                }
            }
        }
        l3.step();
        assert!(l3.population() <= 12 * 12 * 12);
        assert!(LifeLike3D::from_rule_string(12, 12, 12, "B(10)/S(12)(13)").is_ok());
        assert!(LifeLike3D::from_rule_string(12, 12, 12, "B(30)/S1").is_err());
    }

    #[test]
    fn test_totalistic_and_sandpile() {
        let rule = totalistic_rule(3, 1815); // 3-state code
        assert_eq!(rule(&[0, 0, 0]), (1815 % 3) as u8);
        assert_eq!(rule(&[1, 1, 0]), ((1815 / 9) % 3) as u8);
        // Sandpile: a big center pile relaxes into a stable state.
        let (w, h) = (21, 21);
        let mut grid = vec![0u32; w * h];
        grid[(h / 2) * w + w / 2] = 1000;
        let topples = sandpile_abelian(&mut grid, w, h);
        assert!(topples > 200, "large pile topples many times");
        assert!(grid.iter().all(|&g| g < 4), "stable configuration");
        // The identity is idempotent: identity + identity stabilizes
        // back to the identity.
        let id = sandpile_identity(9, 9);
        assert!(id.iter().all(|&g| g < 4));
        let mut doubled: Vec<u32> = id.iter().map(|&g| g * 2).collect();
        sandpile_abelian(&mut doubled, 9, 9);
        assert_eq!(doubled, id, "sandpile identity is the group identity");
    }

    #[test]
    fn test_stochastic_lattice_models() {
        let mut rng = Rng::new(9);
        // Forest fire: states stay in {0, 1, 2} and trees persist.
        let history = forest_fire(24, 24, 0.05, 0.001, 30, &mut rng);
        assert_eq!(history.len(), 31);
        assert!(history.iter().all(|g| g.iter().all(|&c| c <= 2)));
        // Greenberg-Hastings waves stay in range.
        let gh = greenberg_hastings(24, 24, 5, 20, &mut rng);
        assert!(gh.iter().all(|g| g.iter().all(|&c| c < 5)));
        // Majority rule coarsens: flips decrease.
        let mut cells: Vec<bool> = (0..32 * 32).map(|_| rng.next_f64() < 0.5).collect();
        let before = cells.clone();
        majority_rule(&mut cells, 32, 32, 8);
        let mut probe = cells.clone();
        majority_rule(&mut probe, 32, 32, 1);
        let late_flips = probe.iter().zip(&cells).filter(|(a, b)| a != b).count();
        let mut probe0 = before.clone();
        majority_rule(&mut probe0, 32, 32, 1);
        let early_flips = probe0.iter().zip(&before).filter(|(a, b)| a != b).count();
        assert!(late_flips < early_flips, "majority dynamics settle ({early_flips} -> {late_flips})");
        // Voter model conserves the state alphabet.
        voter_model(&mut cells, 32, 32, 2000, &mut rng);
        // Schelling: segregation index rises above the mixed 0.5.
        let mut grid: Vec<u8> = (0..32 * 32)
            .map(|_| {
                let r = rng.next_f64();
                if r < 0.45 {
                    1
                } else if r < 0.9 {
                    2
                } else {
                    0
                }
            })
            .collect();
        let index = schelling_segregation(&mut grid, 32, 32, 0.5, 40, &mut rng);
        assert!(index > 0.7, "agents segregate ({index})");
    }

    #[test]
    fn test_gray_scott_bounded_and_patterns() {
        let mut gs = GrayScott::coral(48, 48);
        gs.seed_square(20, 20, 6);
        gs.run(300);
        for (&u, &v) in gs.u.iter().zip(&gs.v) {
            assert!((0.0..=1.5).contains(&u), "u bounded ({u})");
            assert!((0.0..=1.5).contains(&v), "v bounded ({v})");
        }
        // The seed has spread structure: v is non-zero away from the
        // original square but not everywhere.
        let active = gs.v.iter().filter(|&&v| v > 0.1).count();
        assert!(active > 36, "pattern grew ({active})");
        assert!(active < 48 * 48 / 2, "pattern is structured");
        // Presets exist with the Pearson parameters.
        assert_eq!(GrayScott::mitosis(8, 8).feed, 0.0367);
        assert_eq!(GrayScott::waves(8, 8).kill, 0.045);
        for p in [
            GrayScott::spots(8, 8),
            GrayScott::worms(8, 8),
            GrayScott::maze(8, 8),
            GrayScott::holes(8, 8),
            GrayScott::solitons(8, 8),
        ] {
            assert!(p.feed > 0.0 && p.kill > 0.0);
        }
    }

    #[test]
    fn test_fhn_spiral_rotates() {
        let mut fhn = FitzHughNagumo::new(48, 48);
        fhn.spiral_wave_seed();
        // Track the sign of v at a probe point: a rotating spiral
        // drives repeated oscillations.
        let probe = 30 * 48 + 30;
        let mut crossings = 0;
        let mut last = fhn.v[probe] > 0.0;
        for _ in 0..5000 {
            fhn.step();
            let now = fhn.v[probe] > 0.0;
            if now != last {
                crossings += 1;
                last = now;
            }
        }
        assert!(crossings >= 8, "spiral wave passes the probe repeatedly ({crossings})");
        assert!(fhn.v.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_other_reaction_diffusion() {
        let mut rng = Rng::new(3);
        let mut turing = Turing::new(24, 24, &mut rng);
        for _ in 0..200 {
            turing.step();
        }
        assert!(turing.activator.iter().all(|a| a.is_finite() && *a >= 0.0));
        let mut bz = BelousovZhabotinsky::new(24, 24);
        for _ in 0..200 {
            bz.step();
        }
        assert!(bz.u.iter().all(|u| u.is_finite()));
        let mut br = Brusselator::new(16, 16, 1.0, 3.0, &mut rng);
        for _ in 0..200 {
            br.step();
        }
        assert!(br.u.iter().all(|u| u.is_finite() && *u >= 0.0));
        // 1-D RD: pure diffusion decays toward the mean.
        let mut u: Vec<f64> = (0..32).map(|i| if i == 16 { 1.0 } else { 0.0 }).collect();
        let mut v = vec![0.0; 32];
        reaction_diffusion_1d(&mut u, &mut v, &|_, _| (0.0, 0.0), 0.2, 0.2, 0.1, 1.0, 200);
        let max = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max < 0.5, "diffusion spreads the spike ({max})");
        let total: f64 = u.iter().sum();
        assert!(total > 0.5, "mass roughly conserved ({total})");
    }

    #[test]
    fn test_smoothlife_and_lenia_bounded() {
        let mut rng = Rng::new(11);
        let mut sl = SmoothLife::new(32, 32, SmoothLifeParams::default());
        for f in sl.field.iter_mut() {
            *f = rng.next_f64();
        }
        for _ in 0..5 {
            sl.step(0.3);
            assert!(sl.field.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        let mut lenia = Lenia::new(40, 40, 6, 0.15, 0.017);
        for (i, f) in lenia.field.iter_mut().enumerate() {
            let (x, y) = ((i % 40) as f64, (i / 40) as f64);
            let r2 = (x - 20.0) * (x - 20.0) + (y - 20.0) * (y - 20.0);
            *f = (-r2 / 30.0).exp();
        }
        for _ in 0..5 {
            lenia.step(0.1);
            assert!(lenia.field.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        let mass: f64 = lenia.field.iter().sum();
        assert!(mass > 0.0, "the blob survives a few steps");
    }

    #[test]
    fn test_dla_eden_invasion() {
        let mut rng = Rng::new(21);
        let cluster = diffusion_limited_aggregation(101, 101, 2500, 1.0, &mut rng);
        let mass = cluster.iter().filter(|&&c| c).count();
        // Growth stops when the cluster reaches the kill boundary, so
        // the mass is capped by the geometry rather than `particles`.
        assert!(mass > 700, "cluster grew to the boundary ({mass})");
        let d = dla_fractal_dimension(&cluster, 101, 101);
        assert!((d - 1.71).abs() < 0.15, "DLA dimension {d} vs 1.71");
        // Eden growth is compact: dimension ~2.
        let eden = eden_growth(64, 64, 1200, &mut rng);
        let de = dla_fractal_dimension(&eden, 64, 64);
        assert!(de > 1.85, "Eden cluster is compact ({de})");
        // Invasion percolation reaches the boundary.
        let inv = invasion_percolation(48, 48, &mut rng);
        let (labels, _) = percolation_cluster(&inv, 48, 48);
        assert!(labels.iter().any(|&l| l > 0));
        let touches_edge = (0..48).any(|x| {
            inv[x] || inv[47 * 48 + x] || inv[x * 48] || inv[x * 48 + 47]
        });
        assert!(touches_edge, "invasion reached a boundary");
    }

    #[test]
    fn test_percolation_threshold() {
        let mut rng = Rng::new(33);
        let p = percolation_threshold_estimate(64, 64, 60, &mut rng);
        assert!((p - 0.5927).abs() < 0.02, "site percolation threshold {p}");
        // Cluster labeling: two disjoint clusters get distinct labels.
        let mut grid = vec![false; 25];
        grid[0] = true;
        grid[1] = true;
        grid[24] = true;
        let (labels, spans) = percolation_cluster(&grid, 5, 5);
        assert_ne!(labels[0], 0);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[24]);
        assert!(!spans);
        // A full column spans.
        let mut col = vec![false; 25];
        for y in 0..5 {
            col[y * 5 + 2] = true;
        }
        assert!(percolation_cluster(&col, 5, 5).1);
    }
}
