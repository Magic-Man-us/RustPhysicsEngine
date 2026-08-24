//! Quasi-random (low-discrepancy) sequences: Sobol and Halton.
//!
//! Sobol points use the Gray-code construction of Bratley & Fox with
//! the Joe-Kuo (new-joe-kuo-6) primitive polynomials and initial
//! direction numbers, embedded here for dimensions up to 21. Halton
//! points use the radical inverse in the first `dim` primes.

use crate::monte_carlo::Rng;

const SOBOL_MAX_DIM: usize = 21;
const SOBOL_BITS: u32 = 32;

// Joe-Kuo new-joe-kuo-6 parameters for dimensions 2..=21:
// (s, a, m[0..s]) — degree, polynomial coefficient bits, initial
// direction numbers. Dimension 1 is the van der Corput sequence.
const JOE_KUO: [(u32, u32, [u32; 7]); 20] = [
    (1, 0, [1, 0, 0, 0, 0, 0, 0]),
    (2, 1, [1, 3, 0, 0, 0, 0, 0]),
    (3, 1, [1, 3, 1, 0, 0, 0, 0]),
    (3, 2, [1, 1, 1, 0, 0, 0, 0]),
    (4, 1, [1, 1, 3, 3, 0, 0, 0]),
    (4, 4, [1, 3, 5, 13, 0, 0, 0]),
    (5, 2, [1, 1, 5, 5, 17, 0, 0]),
    (5, 4, [1, 1, 5, 5, 5, 0, 0]),
    (5, 7, [1, 1, 7, 11, 19, 0, 0]),
    (5, 11, [1, 1, 5, 1, 1, 0, 0]),
    (5, 13, [1, 1, 1, 3, 11, 0, 0]),
    (5, 14, [1, 3, 5, 5, 31, 0, 0]),
    (6, 1, [1, 3, 3, 9, 7, 49, 0]),
    (6, 13, [1, 1, 1, 15, 21, 21, 0]),
    (6, 16, [1, 3, 1, 13, 27, 49, 0]),
    (6, 19, [1, 1, 1, 15, 7, 5, 0]),
    (6, 22, [1, 3, 1, 15, 13, 25, 0]),
    (6, 25, [1, 1, 5, 5, 19, 61, 0]),
    (7, 1, [1, 3, 7, 11, 23, 15, 103]),
    (7, 4, [1, 3, 7, 13, 13, 15, 69]),
];

/// Sobol low-discrepancy sequence in `dim` dimensions (dim ≤ 21).
///
/// Successive calls to [`Sobol::next`] return points x₁, x₂, … in
/// [0, 1)^dim (the origin point x₀ = 0 is skipped).
pub struct Sobol {
    dim: usize,
    index: u64,
    /// Direction integers per dimension, scaled to 2³² (index k holds
    /// v_k for bit position k).
    direction: Vec<Vec<u64>>,
    /// Current Gray-code state per dimension (integer numerators).
    state: Vec<u64>,
}

impl Sobol {
    /// # Panics
    /// Panics unless 1 ≤ dim ≤ 21.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        assert!(
            (1..=SOBOL_MAX_DIM).contains(&dim),
            "Sobol supports 1..=21 dimensions (embedded Joe-Kuo table)"
        );
        let bits = SOBOL_BITS as usize;
        let mut direction = Vec::with_capacity(dim);
        // Dimension 1: van der Corput, v_k = 2^{32-k-1}.
        let mut v0 = vec![0u64; bits];
        for (k, v) in v0.iter_mut().enumerate() {
            *v = 1u64 << (SOBOL_BITS as usize - k - 1);
        }
        direction.push(v0);
        for d in 1..dim {
            let (s, a, m) = JOE_KUO[d - 1];
            let s = s as usize;
            let mut v = vec![0u64; bits];
            for k in 0..bits {
                if k < s {
                    v[k] = (m[k] as u64) << (SOBOL_BITS as usize - k - 1);
                } else {
                    // Recurrence: v_k = v_{k-s} ^ (v_{k-s} >> s) ^ Σ a-bits v_{k-i}.
                    let mut val = v[k - s] ^ (v[k - s] >> s);
                    for i in 1..s {
                        if (a >> (s - 1 - i)) & 1 == 1 {
                            val ^= v[k - i];
                        }
                    }
                    v[k] = val;
                }
            }
            direction.push(v);
        }
        Self { dim, index: 0, direction, state: vec![0; dim] }
    }

    /// The next point in the sequence.
    ///
    /// Named `next` to match the sequence vocabulary; it is deliberately an
    /// inherent method rather than `Iterator`, which cannot borrow `self`
    /// mutably and yield a fresh `Vec` without allocating an adapter.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn next(&mut self) -> Vec<f64> {
        // Gray-code update using the lowest zero bit of the previous index.
        let c = self.index.trailing_ones() as usize;
        self.index += 1;
        let scale = 1.0 / (1u64 << SOBOL_BITS) as f64;
        (0..self.dim)
            .map(|d| {
                self.state[d] ^= self.direction[d][c];
                self.state[d] as f64 * scale
            })
            .collect()
    }

    /// Skips the next n points.
    pub fn skip(&mut self, n: u64) {
        for _ in 0..n {
            let _ = self.next();
        }
    }

    /// Dimensionality of the sequence.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Current integer state (used by the scrambler).
    fn state_scaled(&self, masks: &[u64]) -> Vec<f64> {
        let scale = 1.0 / (1u64 << SOBOL_BITS) as f64;
        self.state
            .iter()
            .zip(masks)
            .map(|(&s, &m)| ((s ^ m) as f64) * scale)
            .collect()
    }
}

const HALTON_PRIMES: [u64; 25] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97,
];

/// Halton low-discrepancy sequence: dimension d uses the radical
/// inverse in the d-th prime.
pub struct Halton {
    dim: usize,
    index: u64,
}

impl Halton {
    /// # Panics
    /// Panics unless 1 ≤ dim ≤ 25.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        assert!((1..=HALTON_PRIMES.len()).contains(&dim), "Halton supports 1..=25 dimensions");
        Self { dim, index: 0 }
    }

    fn radical_inverse(mut n: u64, base: u64) -> f64 {
        let mut inv = 0.0;
        let mut denom = 1.0;
        while n > 0 {
            denom *= base as f64;
            inv += (n % base) as f64 / denom;
            n /= base;
        }
        inv
    }

    /// The next point in the sequence (index starts at 1).
    ///
    /// Inherent rather than `Iterator` for the same reason as the Sobol
    /// sequence above.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn next(&mut self) -> Vec<f64> {
        self.index += 1;
        (0..self.dim)
            .map(|d| Self::radical_inverse(self.index, HALTON_PRIMES[d]))
            .collect()
    }
}

/// Quasi-Monte Carlo integration of f over [0, 1]^dim with the first n
/// Sobol points.
///
/// # Panics
/// Panics unless n > 0 and dim is Sobol-supported.
#[must_use]
pub fn mc_integrate_sobol(f: &dyn Fn(&[f64]) -> f64, dim: usize, n: usize) -> f64 {
    assert!(n > 0, "mc_integrate_sobol requires n > 0");
    let mut seq = Sobol::new(dim);
    let mut sum = 0.0;
    for _ in 0..n {
        sum += f(&seq.next());
    }
    sum / n as f64
}

/// Digitally shifted (random-XOR scrambled) Sobol iterator: each
/// dimension's integer state is XORed with a fixed random 32-bit mask,
/// preserving the net's equidistribution while randomizing the points
/// (Cranley-Patterson-style digital shift; a lightweight stand-in for
/// full Owen scrambling).
pub fn scrambled(mut seq: Sobol, rng: &mut Rng) -> impl Iterator<Item = Vec<f64>> {
    let masks: Vec<u64> = (0..seq.dim()).map(|_| rng.next_u64() >> SOBOL_BITS).collect();
    std::iter::from_fn(move || {
        let _ = seq.next();
        Some(seq.state_scaled(&masks))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobol_first_points_dim1() {
        // Van der Corput in base 2 (Gray-code order): 1/2, 3/4, 1/4, 3/8, ...
        let mut s = Sobol::new(1);
        assert_eq!(s.next()[0], 0.5);
        assert_eq!(s.next()[0], 0.75);
        assert_eq!(s.next()[0], 0.25);
        assert_eq!(s.next()[0], 0.375);
    }

    #[test]
    fn test_sobol_dim2_balance() {
        // The net x0..x15 is balanced per axis; with the origin x0
        // skipped, x1..x15 has exactly 7 low-half points per axis.
        let mut s = Sobol::new(2);
        let pts: Vec<Vec<f64>> = (0..15).map(|_| s.next()).collect();
        for d in 0..2 {
            let low = pts.iter().filter(|p| p[d] < 0.5).count();
            assert_eq!(low, 7, "dimension {d} unbalanced");
        }
    }

    #[test]
    fn test_sobol_skip_matches_sequential() {
        let mut a = Sobol::new(3);
        let mut b = Sobol::new(3);
        b.skip(7);
        for _ in 0..7 {
            let _ = a.next();
        }
        assert_eq!(a.next(), b.next());
    }

    #[test]
    #[should_panic(expected = "1..=21")]
    fn test_sobol_dim_limit() {
        let _ = Sobol::new(22);
    }

    #[test]
    fn test_halton_first_points() {
        let mut h = Halton::new(2);
        let p1 = h.next();
        assert!((p1[0] - 0.5).abs() < 1e-15); // base 2: 1 -> 0.5
        assert!((p1[1] - 1.0 / 3.0).abs() < 1e-15); // base 3: 1 -> 1/3
        let p2 = h.next();
        assert!((p2[0] - 0.25).abs() < 1e-15);
        assert!((p2[1] - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_sobol_integration_product() {
        // ∫ over [0,1]^4 of Π x_i = 1/16.
        let f = |x: &[f64]| x.iter().product::<f64>();
        let est = mc_integrate_sobol(&f, 4, 4096);
        assert!((est - 1.0 / 16.0).abs() < 1e-4, "estimate {est}");
    }

    #[test]
    fn test_scrambled_stays_in_unit_cube() {
        let mut rng = Rng::new(101);
        let it = scrambled(Sobol::new(3), &mut rng);
        for p in it.take(100) {
            assert!(p.iter().all(|&x| (0.0..1.0).contains(&x)));
        }
    }
}
