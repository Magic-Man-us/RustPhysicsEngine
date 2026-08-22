//! Compensated (error-free-transformation) summation.
//!
//! Formulas: Neumaier's improved Kahan-Babuska summation
//! (A. Neumaier, "Rundungsfehleranalyse einiger Verfahren zur Summation
//! endlicher Summen", ZAMM 54, 1974) and recursive pairwise summation
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4).

/// Neumaier compensated sum of a slice.
///
/// Computes `Σ xᵢ` with a running compensation term that captures the
/// low-order bits lost in each addition, giving results accurate to
/// O(1) ulp independent of length for well-scaled data.
#[must_use]
pub fn sum_neumaier(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64; // running compensation for lost low-order bits
    for &x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Recursive pairwise sum of a slice: `Σ xᵢ` with O(log n) error growth.
///
/// Splits the slice in half and sums each half recursively; runs of up
/// to 32 elements are summed naively as the base case.
#[must_use]
pub fn sum_pairwise(xs: &[f64]) -> f64 {
    const BASE: usize = 32;
    if xs.len() <= BASE {
        return xs.iter().sum();
    }
    let mid = xs.len() / 2;
    sum_pairwise(&xs[..mid]) + sum_pairwise(&xs[mid..])
}

/// Compensated dot product `Σ aᵢ·bᵢ` via Neumaier accumulation of the
/// individual products.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn dot_compensated(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() == b.len(), "dot_compensated requires equal-length slices");
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let p = x * y;
        let t = sum + p;
        if sum.abs() >= p.abs() {
            comp += (sum - t) + p;
        } else {
            comp += (p - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_neumaier_exact_small() {
        assert_eq!(sum_neumaier(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(sum_neumaier(&[]), 0.0);
    }

    #[test]
    fn test_sum_neumaier_catastrophic_case() {
        // 1 + 1e100 - 1e100 + 1: naive sum gives 0, compensated gives 2.
        let xs = [1.0, 1e100, 1.0, -1e100];
        assert_eq!(sum_neumaier(&xs), 2.0);
    }

    #[test]
    fn test_sum_neumaier_many_tenths() {
        let xs = vec![0.1_f64; 1_000_000];
        assert!((sum_neumaier(&xs) - 1e5).abs() < 1e-9);
    }

    #[test]
    fn test_sum_pairwise_matches_exact() {
        let xs: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        assert_eq!(sum_pairwise(&xs), 500_500.0);
        assert_eq!(sum_pairwise(&[]), 0.0);
        assert_eq!(sum_pairwise(&[4.5]), 4.5);
    }

    #[test]
    fn test_sum_pairwise_better_than_naive() {
        let xs = vec![0.1_f64; 1_000_000];
        let naive: f64 = xs.iter().sum();
        let pairwise = sum_pairwise(&xs);
        assert!((pairwise - 1e5).abs() < (naive - 1e5).abs());
    }

    #[test]
    fn test_dot_compensated() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot_compensated(&a, &b), 32.0);
    }

    #[test]
    #[should_panic(expected = "equal-length")]
    fn test_dot_compensated_length_mismatch() {
        let _ = dot_compensated(&[1.0], &[1.0, 2.0]);
    }
}
