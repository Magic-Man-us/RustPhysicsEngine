//! Rigorous interval arithmetic with outward rounding.
//!
//! Every operation returns an interval guaranteed to contain the true
//! real result for all inputs in the operand intervals: computed bounds
//! are widened outward with `f64::next_down`/`next_up` (plus a small
//! ulp margin for transcendental functions whose libm error is ≤ 1 ulp
//! but unproven). Reference: Moore, Kearfott & Cloud, *Introduction to
//! Interval Analysis* (SIAM, 2009).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Closed interval [lo, hi].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

/// Widens a computed lower bound downward by two ulps (covers a ≤ 1 ulp
/// evaluation error plus the rounding of the widening itself).
fn down2(x: f64) -> f64 {
    x.next_down().next_down()
}

fn up2(x: f64) -> f64 {
    x.next_up().next_up()
}

impl Interval {
    /// # Panics
    /// Panics unless lo ≤ hi and both are finite.
    #[must_use]
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "Interval requires lo <= hi");
        assert!(lo.is_finite() && hi.is_finite(), "Interval requires finite bounds");
        Self { lo, hi }
    }

    /// Degenerate interval [x, x].
    ///
    /// # Panics
    /// Panics unless x is finite.
    #[must_use]
    pub fn point(x: f64) -> Self {
        Self::new(x, x)
    }

    /// Width hi − lo.
    #[must_use]
    pub fn width(self) -> f64 {
        self.hi - self.lo
    }

    /// Midpoint (lo + hi)/2.
    #[must_use]
    pub fn midpoint(self) -> f64 {
        0.5 * (self.lo + self.hi)
    }

    /// True when x ∈ [lo, hi].
    #[must_use]
    pub fn contains(self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// Intersection, or `None` when the intervals are disjoint.
    #[must_use]
    pub fn intersect(self, other: Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi {
            Some(Self { lo, hi })
        } else {
            None
        }
    }

    /// Smallest interval containing both operands.
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    /// Interval square root; the operand must be non-negative.
    ///
    /// # Panics
    /// Panics if lo < 0.
    #[must_use]
    pub fn sqrt(self) -> Self {
        assert!(self.lo >= 0.0, "Interval::sqrt requires a non-negative interval");
        Self { lo: down2(self.lo.sqrt()).max(0.0), hi: up2(self.hi.sqrt()) }
    }

    /// Interval exponential.
    #[must_use]
    pub fn exp(self) -> Self {
        Self { lo: down2(self.lo.exp()).max(0.0), hi: up2(self.hi.exp()) }
    }

    /// Interval sine: exact range over the interval, endpoints widened
    /// outward.
    #[must_use]
    pub fn sin(self) -> Self {
        self.trig_range(f64::sin, std::f64::consts::FRAC_PI_2)
    }

    /// Interval cosine.
    #[must_use]
    pub fn cos(self) -> Self {
        self.trig_range(f64::cos, 0.0)
    }

    /// Shared machinery: range of a 2π-periodic wave with maxima at
    /// `max_offset + 2πk` and minima at `max_offset + π + 2πk`.
    fn trig_range(self, f: fn(f64) -> f64, max_offset: f64) -> Self {
        use std::f64::consts::{PI, TAU};
        if self.width() >= TAU {
            return Self { lo: -1.0, hi: 1.0 };
        }
        let mut lo = f(self.lo).min(f(self.hi));
        let mut hi = f(self.lo).max(f(self.hi));
        // Does [lo, hi] contain a maximum point max_offset + 2πk?
        let contains_crit = |offset: f64| {
            let k = ((self.lo - offset) / TAU).ceil();
            let crit = offset + k * TAU;
            crit <= self.hi
        };
        if contains_crit(max_offset) {
            hi = 1.0;
        }
        if contains_crit(max_offset + PI) {
            lo = -1.0;
        }
        Self { lo: down2(lo).max(-1.0), hi: up2(hi).min(1.0) }
    }

    /// Integer power with exact monotonicity handling (even powers of
    /// sign-straddling intervals reach down to 0).
    ///
    /// # Panics
    /// Panics for negative n when the interval contains 0.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self { lo: 1.0, hi: 1.0 };
        }
        if n < 0 {
            assert!(
                !self.contains(0.0),
                "Interval::powi with negative exponent requires an interval excluding 0"
            );
            let pos = self.powi(-n);
            return Self { lo: down2(1.0 / pos.hi), hi: up2(1.0 / pos.lo) };
        }
        let pl = self.lo.powi(n);
        let ph = self.hi.powi(n);
        if n % 2 == 1 {
            Self { lo: down2(pl), hi: up2(ph) }
        } else if self.contains(0.0) {
            Self { lo: 0.0, hi: up2(pl.max(ph)) }
        } else {
            Self { lo: down2(pl.min(ph)).max(0.0), hi: up2(pl.max(ph)) }
        }
    }
}

impl Add for Interval {
    type Output = Interval;
    fn add(self, rhs: Interval) -> Interval {
        Interval { lo: (self.lo + rhs.lo).next_down(), hi: (self.hi + rhs.hi).next_up() }
    }
}

impl Sub for Interval {
    type Output = Interval;
    fn sub(self, rhs: Interval) -> Interval {
        Interval { lo: (self.lo - rhs.hi).next_down(), hi: (self.hi - rhs.lo).next_up() }
    }
}

impl Mul for Interval {
    type Output = Interval;
    fn mul(self, rhs: Interval) -> Interval {
        let c = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Interval { lo: lo.next_down(), hi: hi.next_up() }
    }
}

impl Div for Interval {
    type Output = Interval;
    /// # Panics
    /// Panics when the divisor contains 0.
    fn div(self, rhs: Interval) -> Interval {
        assert!(!rhs.contains(0.0), "interval division by an interval containing 0");
        let c = [
            self.lo / rhs.lo,
            self.lo / rhs.hi,
            self.hi / rhs.lo,
            self.hi / rhs.hi,
        ];
        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Interval { lo: lo.next_down(), hi: hi.next_up() }
    }
}

impl Neg for Interval {
    type Output = Interval;
    fn neg(self) -> Interval {
        Interval { lo: -self.hi, hi: -self.lo }
    }
}

const NEWTON_MAX_SUBDIVISIONS: usize = 100_000;

/// Rigorous interval Newton method: encloses every root of f in `x0`.
///
/// `f` and `df` must be interval extensions of the function and its
/// derivative. Boxes where 0 ∉ f(X) are discarded; where f'(X)
/// excludes 0 the Newton contraction N(X) = m − f(m)/f'(X) ∩ X is
/// applied; otherwise the box is bisected. Boxes narrower than `tol`
/// that still satisfy 0 ∈ f(X) are reported (overlapping neighbors
/// merged). Every real root in `x0` is contained in some returned
/// interval; spurious near-root boxes may also appear at width ~tol.
///
/// # Panics
/// Panics unless tol > 0.
#[must_use]
pub fn interval_newton(
    f: &dyn Fn(Interval) -> Interval,
    df: &dyn Fn(Interval) -> Interval,
    x0: Interval,
    tol: f64,
    max_iter: usize,
) -> Vec<Interval> {
    assert!(tol > 0.0, "interval_newton requires tol > 0");
    let mut work = vec![x0];
    let mut found: Vec<Interval> = Vec::new();
    let mut budget = max_iter.max(1) * 64;
    if budget > NEWTON_MAX_SUBDIVISIONS {
        budget = NEWTON_MAX_SUBDIVISIONS;
    }
    while let Some(x) = work.pop() {
        if budget == 0 {
            // Conservative: report the unresolved box rather than lose roots.
            found.push(x);
            continue;
        }
        budget -= 1;
        let fx = f(x);
        if !fx.contains(0.0) {
            continue;
        }
        if x.width() < tol {
            found.push(x);
            continue;
        }
        let m = x.midpoint();
        let dfx = df(x);
        if dfx.contains(0.0) {
            // Derivative straddles zero: bisect.
            work.push(Interval::new(x.lo, m));
            work.push(Interval::new(m, x.hi));
            continue;
        }
        let n = Interval::point(m) - f(Interval::point(m)) / dfx;
        match n.intersect(x) {
            None => continue, // no root in this box
            Some(xn) => {
                if xn.width() > 0.9 * x.width() {
                    // Slow contraction: bisect instead.
                    work.push(Interval::new(x.lo, m));
                    work.push(Interval::new(m, x.hi));
                } else {
                    work.push(xn);
                }
            }
        }
    }
    // Merge overlapping/adjacent boxes.
    found.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<Interval> = Vec::new();
    for iv in found {
        match merged.last_mut() {
            Some(last) if iv.lo <= last.hi => {
                last.hi = last.hi.max(iv.hi);
            }
            _ => merged.push(iv),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_contains_exact() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(-3.0, 0.5);
        let s = a + b;
        assert!(s.lo <= -2.0 && s.hi >= 2.5);
        let d = a - b;
        assert!(d.lo <= 0.5 && d.hi >= 5.0);
        let p = a * b;
        assert!(p.lo <= -6.0 && p.hi >= 1.0);
        let q = a / Interval::new(2.0, 4.0);
        assert!(q.lo <= 0.25 && q.hi >= 1.0);
        let n = -a;
        assert_eq!((n.lo, n.hi), (-2.0, -1.0));
    }

    #[test]
    #[should_panic(expected = "containing 0")]
    fn test_division_by_zero_interval_panics() {
        let _ = Interval::new(1.0, 2.0) / Interval::new(-1.0, 1.0);
    }

    #[test]
    fn test_point_width_contains_intersect_hull() {
        let p = Interval::point(3.0);
        assert_eq!(p.width(), 0.0);
        assert!(p.contains(3.0));
        let a = Interval::new(0.0, 2.0);
        let b = Interval::new(1.0, 5.0);
        let i = a.intersect(b).unwrap();
        assert_eq!((i.lo, i.hi), (1.0, 2.0));
        assert!(a.intersect(Interval::new(3.0, 4.0)).is_none());
        let h = a.hull(b);
        assert_eq!((h.lo, h.hi), (0.0, 5.0));
    }

    #[test]
    fn test_sqrt_exp() {
        let a = Interval::new(4.0, 9.0);
        let r = a.sqrt();
        assert!(r.lo <= 2.0 && r.hi >= 3.0);
        assert!(r.lo > 1.99 && r.hi < 3.01);
        let e = Interval::new(0.0, 1.0).exp();
        assert!(e.lo <= 1.0 && e.hi >= std::f64::consts::E);
    }

    // Miri evaluates the float intrinsics with its own implementations, which
    // are allowed to differ from the host's in the last bits and which Miri
    // deliberately randomises within that slack. This test asserts an exact
    // value, so it fails under Miri for that reason and not because anything
    // is wrong; it still runs normally everywhere else.
    #[cfg_attr(miri, ignore = "Miri's float intrinsics are not bit-exact")]
    #[test]
    fn test_sin_ranges() {
        use std::f64::consts::PI;
        // Interval through the maximum at pi/2.
        let s = Interval::new(0.0, PI).sin();
        assert!(s.hi >= 1.0);
        assert!(s.lo <= 0.0 && s.lo > -1e-10);
        // Narrow interval on a monotone stretch.
        let s2 = Interval::new(0.1, 0.2).sin();
        assert!(s2.lo <= 0.1_f64.sin() && s2.hi >= 0.2_f64.sin());
        assert!(s2.width() < 0.11);
        // Full period covers [-1, 1].
        let s3 = Interval::new(0.0, 7.0).sin();
        assert_eq!((s3.lo, s3.hi), (-1.0, 1.0));
        // Cosine maximum at 0/2pi.
        let c = Interval::new(-0.5, 0.5).cos();
        assert!(c.hi >= 1.0);
    }

    // Miri evaluates the float intrinsics with its own implementations, which
    // are allowed to differ from the host's in the last bits and which Miri
    // deliberately randomises within that slack. This test asserts an exact
    // value, so it fails under Miri for that reason and not because anything
    // is wrong; it still runs normally everywhere else.
    #[cfg_attr(miri, ignore = "Miri's float intrinsics are not bit-exact")]
    #[test]
    fn test_powi_cases() {
        let a = Interval::new(-2.0, 3.0);
        let sq = a.powi(2);
        assert!(sq.lo <= 0.0 && sq.hi >= 9.0);
        let cu = a.powi(3);
        assert!(cu.lo <= -8.0 && cu.hi >= 27.0);
        let inv = Interval::new(2.0, 4.0).powi(-1);
        assert!(inv.lo <= 0.25 && inv.hi >= 0.5);
        let one = a.powi(0);
        assert_eq!((one.lo, one.hi), (1.0, 1.0));
    }

    #[test]
    fn test_interval_newton_finds_both_sqrt2_roots() {
        // f(x) = x^2 - 2 on [-3, 3]: roots ±sqrt(2).
        let f = |x: Interval| x * x - Interval::point(2.0);
        let df = |x: Interval| Interval::point(2.0) * x;
        let roots = interval_newton(&f, &df, Interval::new(-3.0, 3.0), 1e-10, 200);
        assert!(!roots.is_empty());
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(
            roots.iter().any(|r| r.contains(sqrt2)),
            "missing +sqrt2 in {roots:?}"
        );
        assert!(
            roots.iter().any(|r| r.contains(-sqrt2)),
            "missing -sqrt2 in {roots:?}"
        );
        for r in &roots {
            assert!(r.width() < 1e-8);
        }
    }

    #[test]
    fn test_interval_newton_no_roots() {
        let f = |x: Interval| x * x + Interval::point(1.0);
        let df = |x: Interval| Interval::point(2.0) * x;
        let roots = interval_newton(&f, &df, Interval::new(-5.0, 5.0), 1e-10, 200);
        assert!(roots.is_empty(), "unexpected roots: {roots:?}");
    }

    #[test]
    fn test_interval_newton_transcendental() {
        // sin(x) = 0 on [2, 4]: the only root is pi.
        let f = |x: Interval| x.sin();
        let df = |x: Interval| x.cos();
        let roots = interval_newton(&f, &df, Interval::new(2.0, 4.0), 1e-12, 200);
        assert_eq!(roots.len(), 1, "roots: {roots:?}");
        assert!(roots[0].contains(std::f64::consts::PI));
    }
}
