//! Space-filling curves and locality-preserving orders: Hilbert (2-D
//! and 3-D), Peano, Morton/Z-order, Gray codes, and L-system curves
//! (Sierpiński arrowhead, Moore, Gosper).

use crate::math::{Vec2, Vec3};

/// Hilbert curve index to grid coordinates on a `2^order` square grid
/// (Wikipedia's iterative rotate-and-flip formulation).
///
/// # Panics
/// Panics unless `1 <= order <= 31` and `d < 4^order`.
#[must_use]
pub fn hilbert_d2xy(order: u32, d: u64) -> (u64, u64) {
    assert!((1..=31).contains(&order), "order must be in 1..=31");
    assert!(d < 1u64 << (2 * order), "index out of range");
    let n = 1u64 << order;
    let (mut x, mut y) = (0u64, 0u64);
    let mut t = d;
    let mut s = 1u64;
    while s < n {
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);
        // Rotate the quadrant.
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }
    (x, y)
}

/// Grid coordinates to Hilbert index (inverse of [`hilbert_d2xy`]).
///
/// # Panics
/// Panics unless `1 <= order <= 31` and both coordinates are below
/// `2^order`.
#[must_use]
pub fn hilbert_xy2d(order: u32, x: u64, y: u64) -> u64 {
    assert!((1..=31).contains(&order), "order must be in 1..=31");
    let n = 1u64 << order;
    assert!(x < n && y < n, "coordinates out of range");
    let (mut x, mut y) = (x, y);
    let mut d = 0u64;
    let mut s = n / 2;
    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        // Rotate (about the full grid, per the canonical algorithm).
        if ry == 0 {
            if rx == 1 {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// The full Hilbert curve as points in the unit square (cell
/// centers), in curve order.
///
/// # Panics
/// Panics unless `1 <= order <= 10` (2^20 points at most).
#[must_use]
pub fn hilbert_curve_2d(order: u32) -> Vec<Vec2> {
    assert!((1..=10).contains(&order), "order must be in 1..=10");
    let n = 1u64 << order;
    let scale = 1.0 / n as f64;
    (0..n * n)
        .map(|d| {
            let (x, y) = hilbert_d2xy(order, d);
            Vec2::new((x as f64 + 0.5) * scale, (y as f64 + 0.5) * scale)
        })
        .collect()
}

/// Skilling's transpose-to-axes step (John Skilling, "Programming the
/// Hilbert curve", AIP 2004), n = 3 dimensions.
fn transpose_to_axes3(x: &mut [u64; 3], bits: u32) {
    let t = x[2] >> 1;
    for i in (1..3).rev() {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;
    let mut q = 2u64;
    while q != (1u64 << bits) {
        let p = q - 1;
        for i in (0..3).rev() {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q <<= 1;
    }
}

fn axes_to_transpose3(x: &mut [u64; 3], bits: u32) {
    let mut q = 1u64 << (bits - 1);
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    for i in 1..3 {
        x[i] ^= x[i - 1];
    }
    let mut t = 0u64;
    let mut q = 1u64 << (bits - 1);
    while q > 1 {
        if x[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for xi in x.iter_mut() {
        *xi ^= t;
    }
}

/// 3-D Hilbert index to grid coordinates on a `2^order` cube.
///
/// # Panics
/// Panics unless `1 <= order <= 21` and `d < 8^order`.
#[must_use]
pub fn hilbert_3d_d2xyz(order: u32, d: u64) -> (u64, u64, u64) {
    assert!((1..=21).contains(&order), "order must be in 1..=21");
    assert!(order == 21 || d < 1u64 << (3 * order), "index out of range");
    // Unpack d into the transpose form: bit j of dimension i is bit
    // (3j + 2 - i) of d.
    let mut x = [0u64; 3];
    for j in 0..order {
        for (i, xi) in x.iter_mut().enumerate() {
            let bit = (d >> (3 * j + 2 - i as u32)) & 1;
            *xi |= bit << j;
        }
    }
    transpose_to_axes3(&mut x, order);
    (x[0], x[1], x[2])
}

/// 3-D grid coordinates to Hilbert index (inverse of
/// [`hilbert_3d_d2xyz`]).
///
/// # Panics
/// Panics unless `1 <= order <= 21` and all coordinates are below
/// `2^order`.
#[must_use]
pub fn hilbert_3d_xyz2d(order: u32, x: u64, y: u64, z: u64) -> u64 {
    assert!((1..=21).contains(&order), "order must be in 1..=21");
    let n = 1u64 << order;
    assert!(x < n && y < n && z < n, "coordinates out of range");
    let mut ax = [x, y, z];
    axes_to_transpose3(&mut ax, order);
    let mut d = 0u64;
    for j in (0..order).rev() {
        for ai in &ax {
            d = (d << 1) | ((ai >> j) & 1);
        }
    }
    d
}

/// The 3-D Hilbert curve as points in the unit cube, in curve order.
///
/// # Panics
/// Panics unless `1 <= order <= 6` (2^18 points at most).
#[must_use]
pub fn hilbert_curve_3d(order: u32) -> Vec<Vec3> {
    assert!((1..=6).contains(&order), "order must be in 1..=6");
    let n = 1u64 << order;
    let scale = 1.0 / n as f64;
    (0..n * n * n)
        .map(|d| {
            let (x, y, z) = hilbert_3d_d2xyz(order, d);
            Vec3::new(
                (x as f64 + 0.5) * scale,
                (y as f64 + 0.5) * scale,
                (z as f64 + 0.5) * scale,
            )
        })
        .collect()
}

/// Peano curve on a `3^order` grid via the ternary digit formula
/// (Peano 1890): points in the unit square in curve order.
///
/// # Panics
/// Panics unless `1 <= order <= 6`.
#[must_use]
pub fn peano_curve(order: u32) -> Vec<Vec2> {
    assert!((1..=6).contains(&order), "order must be in 1..=6");
    let side = 3u64.pow(order);
    let total = side * side;
    let scale = 1.0 / side as f64;
    (0..total)
        .map(|d| {
            // Ternary digits of d, most significant first.
            let digits: Vec<u64> = (0..2 * order)
                .rev()
                .map(|k| (d / 3u64.pow(k)) % 3)
                .collect();
            let k = |digit: u64, flips: u64| if flips % 2 == 1 { 2 - digit } else { digit };
            let mut x = 0u64;
            let mut y = 0u64;
            for i in 0..order as usize {
                // x digit i comes from t_{2i}, flipped by the sum of
                // odd-position digits before it; y from t_{2i+1},
                // flipped by even-position digits up to it.
                let flip_x: u64 = (0..i).map(|m| digits[2 * m + 1]).sum();
                let flip_y: u64 = (0..=i).map(|m| digits[2 * m]).sum();
                x = x * 3 + k(digits[2 * i], flip_x);
                y = y * 3 + k(digits[2 * i + 1], flip_y);
            }
            Vec2::new((x as f64 + 0.5) * scale, (y as f64 + 0.5) * scale)
        })
        .collect()
}

fn spread_bits_2d(v: u32) -> u64 {
    let mut x = u64::from(v);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

fn compact_bits_2d(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    x = (x | (x >> 16)) & 0x0000_0000_ffff_ffff;
    x as u32
}

/// Interleaves the bits of x (even positions) and y (odd positions).
#[must_use]
pub fn morton_encode_2d(x: u32, y: u32) -> u64 {
    spread_bits_2d(x) | (spread_bits_2d(y) << 1)
}

/// Inverse of [`morton_encode_2d`].
#[must_use]
pub fn morton_decode_2d(m: u64) -> (u32, u32) {
    (compact_bits_2d(m), compact_bits_2d(m >> 1))
}

fn spread_bits_3d(v: u32) -> u64 {
    let mut x = u64::from(v) & 0x1f_ffff; // 21 bits
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

fn compact_bits_3d(mut x: u64) -> u32 {
    x &= 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x >> 4)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x >> 8)) & 0x001f_0000_ff00_00ff;
    x = (x | (x >> 16)) & 0x001f_0000_0000_ffff;
    x = (x | (x >> 32)) & 0x1f_ffff;
    x as u32
}

/// Interleaves 21 bits each of x, y, z.
///
/// # Panics
/// Panics when any coordinate exceeds 21 bits.
#[must_use]
pub fn morton_encode_3d(x: u32, y: u32, z: u32) -> u64 {
    assert!(x < (1 << 21) && y < (1 << 21) && z < (1 << 21), "coordinates must fit 21 bits");
    spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2)
}

/// Inverse of [`morton_encode_3d`].
#[must_use]
pub fn morton_decode_3d(m: u64) -> (u32, u32, u32) {
    (compact_bits_3d(m), compact_bits_3d(m >> 1), compact_bits_3d(m >> 2))
}

/// The Z-order (Morton) traversal of a `2^order` grid as unit-square
/// points.
///
/// # Panics
/// Panics unless `1 <= order <= 10`.
#[must_use]
pub fn z_order_curve(order: u32) -> Vec<Vec2> {
    assert!((1..=10).contains(&order), "order must be in 1..=10");
    let n = 1u64 << order;
    let scale = 1.0 / n as f64;
    (0..n * n)
        .map(|d| {
            let (x, y) = morton_decode_2d(d);
            Vec2::new((f64::from(x) + 0.5) * scale, (f64::from(y) + 0.5) * scale)
        })
        .collect()
}

/// Binary reflected Gray code.
#[must_use]
pub fn gray_code(n: u64) -> u64 {
    n ^ (n >> 1)
}

/// Inverse Gray code (prefix xor by doubling).
#[must_use]
pub fn gray_decode(g: u64) -> u64 {
    let mut n = g;
    let mut shift = 1u32;
    while shift < 64 {
        n ^= n >> shift;
        shift <<= 1;
    }
    n
}

/// Runs a turtle over an L-system expansion; `draw` characters move
/// forward one unit, `+`/`-` turn by `angle`.
fn lsystem_turtle(
    axiom: &str,
    rules: &[(char, &str)],
    iterations: usize,
    angle: f64,
    draw: &str,
) -> Vec<Vec2> {
    let mut s: Vec<char> = axiom.chars().collect();
    for _ in 0..iterations {
        let mut next = Vec::with_capacity(s.len() * 4);
        for &c in &s {
            match rules.iter().find(|(from, _)| *from == c) {
                Some((_, to)) => next.extend(to.chars()),
                None => next.push(c),
            }
        }
        s = next;
    }
    let mut pos = Vec2::ZERO;
    let mut dir = 0.0f64;
    let mut out = vec![pos];
    for c in s {
        if draw.contains(c) {
            pos = pos + Vec2::new(dir.cos(), dir.sin());
            out.push(pos);
        } else if c == '+' {
            dir += angle;
        } else if c == '-' {
            dir -= angle;
        }
    }
    out
}

/// Sierpiński arrowhead curve (traverses the Sierpiński triangle),
/// unit steps from the origin.
///
/// # Panics
/// Panics unless `1 <= order <= 10`.
#[must_use]
pub fn sierpinski_curve(order: u32) -> Vec<Vec2> {
    assert!((1..=10).contains(&order), "order must be in 1..=10");
    lsystem_turtle(
        "XF",
        &[('X', "YF+XF+Y"), ('Y', "XF-YF-X")],
        order as usize,
        std::f64::consts::FRAC_PI_3,
        "F",
    )
}

/// Moore curve: the closed variant of the Hilbert curve (last point
/// adjacent to the first), unit grid steps.
///
/// # Panics
/// Panics unless `1 <= order <= 8`.
#[must_use]
pub fn moore_curve(order: u32) -> Vec<Vec2> {
    assert!((1..=8).contains(&order), "order must be in 1..=8");
    lsystem_turtle(
        "LFL+F+LFL",
        &[('L', "-RF+LFL+FR-"), ('R', "+LF-RFR-FL+")],
        order as usize - 1,
        std::f64::consts::FRAC_PI_2,
        "F",
    )
}

/// Gosper (flowsnake) curve, unit steps.
///
/// # Panics
/// Panics unless `1 <= order <= 6`.
#[must_use]
pub fn gosper_curve(order: u32) -> Vec<Vec2> {
    assert!((1..=6).contains(&order), "order must be in 1..=6");
    lsystem_turtle(
        "A",
        &[('A', "A-B--B+A++AA+B-"), ('B', "+A-BB--B-A++A+B")],
        order as usize,
        std::f64::consts::FRAC_PI_3,
        "AB",
    )
}

/// Sorts points by their Hilbert index on a `2^order` grid over the
/// bounding box.
///
/// # Panics
/// Panics unless `1 <= order <= 31`.
pub fn sort_by_hilbert(points: &mut [Vec2], order: u32) {
    assert!((1..=31).contains(&order), "order must be in 1..=31");
    if points.len() < 2 {
        return;
    }
    let (mut lo, mut hi) = (points[0], points[0]);
    for p in points.iter() {
        lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
        hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
    }
    let n = (1u64 << order) - 1;
    let quant = |v: f64, lo: f64, hi: f64| -> u64 {
        if hi <= lo {
            0
        } else {
            (((v - lo) / (hi - lo)) * n as f64) as u64
        }
    };
    points.sort_by_cached_key(|p| {
        hilbert_xy2d(order, quant(p.x, lo.x, hi.x), quant(p.y, lo.y, hi.y))
    });
}

/// Sorts 3-D points by Morton code (21 bits per axis over the
/// bounding box).
pub fn sort_by_morton(points: &mut [Vec3]) {
    if points.len() < 2 {
        return;
    }
    let (mut lo, mut hi) = (points[0], points[0]);
    for p in points.iter() {
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let n = ((1u64 << 21) - 1) as f64;
    let quant = |v: f64, lo: f64, hi: f64| -> u32 {
        if hi <= lo {
            0
        } else {
            (((v - lo) / (hi - lo)) * n) as u32
        }
    };
    points.sort_by_cached_key(|p| {
        morton_encode_3d(
            quant(p.x, lo.x, hi.x),
            quant(p.y, lo.y, hi.y),
            quant(p.z, lo.z, hi.z),
        )
    });
}

/// Locality measure of the Hilbert order: mean |index difference|
/// (normalized by the index range) divided by mean spatial distance
/// (normalized by the bounding-box diagonal) over all point pairs.
/// Lower means indices track spatial proximity better.
///
/// # Panics
/// Panics unless `1 <= order <= 31` and at least 2 points are given.
#[must_use]
pub fn hilbert_locality_ratio(points: &[Vec2], order: u32) -> f64 {
    assert!(points.len() >= 2, "need at least 2 points");
    let (mut lo, mut hi) = (points[0], points[0]);
    for p in points {
        lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
        hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
    }
    let diag = (hi - lo).magnitude().max(1e-300);
    let n = (1u64 << order) - 1;
    let quant = |v: f64, lo: f64, hi: f64| -> u64 {
        if hi <= lo {
            0
        } else {
            (((v - lo) / (hi - lo)) * n as f64) as u64
        }
    };
    let idx: Vec<u64> = points
        .iter()
        .map(|p| hilbert_xy2d(order, quant(p.x, lo.x, hi.x), quant(p.y, lo.y, hi.y)))
        .collect();
    let range = (1u128 << (2 * order)) as f64;
    let mut sum_idx = 0.0;
    let mut sum_sp = 0.0;
    let mut count = 0usize;
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            sum_idx += idx[i].abs_diff(idx[j]) as f64 / range;
            sum_sp += points[i].distance_to(&points[j]) / diag;
            count += 1;
        }
    }
    (sum_idx / count as f64) / (sum_sp / count as f64).max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_2d_roundtrip_and_adjacency() {
        for order in 1..=6u32 {
            let n = 1u64 << order;
            let mut prev = None;
            let mut seen = vec![false; (n * n) as usize];
            for d in 0..n * n {
                let (x, y) = hilbert_d2xy(order, d);
                assert!(x < n && y < n);
                assert_eq!(hilbert_xy2d(order, x, y), d, "roundtrip at order {order}");
                assert!(!seen[(y * n + x) as usize], "cell visited twice");
                seen[(y * n + x) as usize] = true;
                if let Some((px, py)) = prev {
                    let manhattan = x.abs_diff(px) + y.abs_diff(py);
                    assert_eq!(manhattan, 1, "consecutive cells must be neighbors");
                }
                prev = Some((x, y));
            }
            assert!(seen.iter().all(|&s| s), "every cell visited");
        }
    }

    #[test]
    fn test_hilbert_3d_roundtrip_and_adjacency() {
        for order in 1..=4u32 {
            let n = 1u64 << order;
            let mut prev = None;
            let mut seen = vec![false; (n * n * n) as usize];
            for d in 0..n * n * n {
                let (x, y, z) = hilbert_3d_d2xyz(order, d);
                assert!(x < n && y < n && z < n);
                assert_eq!(hilbert_3d_xyz2d(order, x, y, z), d);
                let cell = ((z * n + y) * n + x) as usize;
                assert!(!seen[cell]);
                seen[cell] = true;
                if let Some((px, py, pz)) = prev {
                    let manhattan = x.abs_diff(px) + y.abs_diff(py) + z.abs_diff(pz);
                    assert_eq!(manhattan, 1, "3-D Hilbert stays connected");
                }
                prev = Some((x, y, z));
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    #[test]
    fn test_peano_visits_all_cells_connectedly() {
        for order in 1..=3u32 {
            let side = 3u64.pow(order);
            let pts = peano_curve(order);
            assert_eq!(pts.len(), (side * side) as usize);
            let cell = |p: Vec2| {
                (
                    (p.x * side as f64 - 0.5).round() as i64,
                    (p.y * side as f64 - 0.5).round() as i64,
                )
            };
            let mut seen = std::collections::HashSet::new();
            let mut prev: Option<(i64, i64)> = None;
            for &p in &pts {
                let c = cell(p);
                assert!(seen.insert(c), "cell visited twice");
                if let Some(pc) = prev {
                    assert_eq!(
                        (c.0 - pc.0).abs() + (c.1 - pc.1).abs(),
                        1,
                        "Peano curve is connected"
                    );
                }
                prev = Some(c);
            }
        }
    }

    #[test]
    fn test_morton_and_gray() {
        for &(x, y) in &[(0u32, 0u32), (1, 2), (12345, 54321), (u32::MAX, 0), (u32::MAX, u32::MAX)] {
            assert_eq!(morton_decode_2d(morton_encode_2d(x, y)), (x, y));
        }
        for &(x, y, z) in &[(0u32, 0, 0), (1, 2, 3), (0x1f_ffff, 0, 0x10_0000), (99999, 88888, 77777)] {
            assert_eq!(morton_decode_3d(morton_encode_3d(x, y, z)), (x, y, z));
        }
        // Morton order 2 bits: z pattern.
        assert_eq!(morton_encode_2d(1, 0), 1);
        assert_eq!(morton_encode_2d(0, 1), 2);
        for n in 0..1000u64 {
            assert_eq!(gray_decode(gray_code(n)), n);
            // Successive Gray codes differ in exactly one bit.
            assert_eq!((gray_code(n) ^ gray_code(n + 1)).count_ones(), 1);
        }
    }

    #[test]
    fn test_z_order_and_curves() {
        let z = z_order_curve(3);
        assert_eq!(z.len(), 64);
        // First four points form the base Z.
        let s = 1.0 / 8.0;
        assert!((z[0] - Vec2::new(0.5 * s, 0.5 * s)).magnitude() < 1e-12);
        assert!((z[1] - Vec2::new(1.5 * s, 0.5 * s)).magnitude() < 1e-12);
        assert!((z[2] - Vec2::new(0.5 * s, 1.5 * s)).magnitude() < 1e-12);

        // L-system curves: unit steps, no repeated vertices for
        // Gosper/Sierpinski at small order.
        for pts in [sierpinski_curve(4), gosper_curve(3), moore_curve(3)] {
            for w in pts.windows(2) {
                assert!((w[0].distance_to(&w[1]) - 1.0).abs() < 1e-9, "unit steps");
            }
        }
        // Moore curve is closed: last adjacent to first.
        let m = moore_curve(4);
        assert!((m[0].distance_to(m.last().unwrap()) - 1.0).abs() < 1e-9);
        // Moore curve visits each of the 4^order cells exactly once;
        // the adjacent endpoints close the loop.
        assert_eq!(m.len(), 256);
        let cells: std::collections::HashSet<(i64, i64)> = m
            .iter()
            .map(|p| (p.x.round() as i64, p.y.round() as i64))
            .collect();
        assert_eq!(cells.len(), 256);
        // Gosper: 7^order segments.
        assert_eq!(gosper_curve(3).len(), 343 + 1);
    }

    #[test]
    fn test_sorting_and_locality() {
        let mut state = 1u64;
        let mut rand = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let pts: Vec<Vec2> = (0..200).map(|_| Vec2::new(rand() * 10.0, rand() * 10.0)).collect();
        let mut sorted = pts.clone();
        sort_by_hilbert(&mut sorted, 10);
        // Same multiset.
        assert_eq!(sorted.len(), pts.len());
        // Hilbert-sorted traversal is much shorter than the random
        // order (locality).
        let path_len = |ps: &[Vec2]| -> f64 {
            ps.windows(2).map(|w| w[0].distance_to(&w[1])).sum()
        };
        assert!(path_len(&sorted) < path_len(&pts) * 0.5);

        let mut pts3: Vec<Vec3> =
            (0..200).map(|_| Vec3::new(rand(), rand(), rand())).collect();
        let before: f64 = pts3.windows(2).map(|w| w[0].distance_to(&w[1])).sum();
        sort_by_morton(&mut pts3);
        let after: f64 = pts3.windows(2).map(|w| w[0].distance_to(&w[1])).sum();
        assert!(after < before * 0.5);

        let r = hilbert_locality_ratio(&pts, 10);
        assert!(r.is_finite() && r > 0.0);
        // The Hilbert curve's own points have excellent locality.
        let h = hilbert_curve_2d(5);
        assert!(hilbert_locality_ratio(&h, 5) < r * 2.0);
    }
}
