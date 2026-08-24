//! Legendre polynomials, associated Legendre functions, real spherical
//! harmonics, and Gauss-Legendre quadrature nodes.
//!
//! References: Abramowitz & Stegun ch. 8; Numerical Recipes §6.8
//! (`plgndr`) and §4.5 (`gauleg`).

use crate::math::constants::PI;

/// Legendre polynomial Pₙ(x) by the Bonnet recurrence
/// (n+1)·P_{n+1} = (2n+1)·x·Pₙ − n·P_{n−1}.
#[must_use]
pub fn legendre_p(n: u32, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p_prev = 1.0;
            let mut p = x;
            for k in 1..n {
                let k = k as f64;
                let p_next = ((2.0 * k + 1.0) * x * p - k * p_prev) / (k + 1.0);
                p_prev = p;
                p = p_next;
            }
            p
        }
    }
}

/// Associated Legendre function Pₗᵐ(x) with the Condon-Shortley phase,
/// for |x| ≤ 1. Negative m uses
/// Pₗ^{−m} = (−1)ᵐ (l−m)!/(l+m)! Pₗᵐ.
///
/// # Panics
/// Panics unless |m| ≤ l and |x| ≤ 1.
#[must_use]
pub fn legendre_p_assoc(l: u32, m: i32, x: f64) -> f64 {
    assert!(m.unsigned_abs() <= l, "legendre_p_assoc requires |m| <= l");
    assert!((-1.0..=1.0).contains(&x), "legendre_p_assoc requires |x| <= 1");
    if m < 0 {
        let ma = m.unsigned_abs();
        let mut ratio = 1.0; // (l-m)!/(l+m)! for m = |m|
        for k in (l - ma + 1)..=(l + ma) {
            ratio /= k as f64;
        }
        let sign = if ma.is_multiple_of(2) { 1.0 } else { -1.0 };
        return sign * ratio * legendre_p_assoc(l, ma as i32, x);
    }
    let m = m as u32;
    // P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let mut fact = 1.0;
        for _ in 0..m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    // P_{m+1}^m = x (2m+1) P_m^m
    let mut pmmp1 = x * (2.0 * m as f64 + 1.0) * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        let ll_f = ll as f64;
        let m_f = m as f64;
        pll = (x * (2.0 * ll_f - 1.0) * pmmp1 - (ll_f + m_f - 1.0) * pmm) / (ll_f - m_f);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Real spherical harmonic Yₗₘ(θ, φ) (orthonormal on the sphere):
/// m > 0 pairs with cos(mφ), m < 0 with sin(|m|φ).
///
/// # Panics
/// Panics unless |m| ≤ l.
#[must_use]
pub fn spherical_harmonic_real(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
    assert!(m.unsigned_abs() <= l, "spherical_harmonic_real requires |m| <= l");
    let ma = m.unsigned_abs();
    // Normalization sqrt((2l+1)/(4 pi) * (l-|m|)!/(l+|m|)!).
    let mut ratio = 1.0;
    for k in (l - ma + 1)..=(l + ma) {
        ratio /= k as f64;
    }
    let norm = ((2.0 * l as f64 + 1.0) / (4.0 * PI) * ratio).sqrt();
    let plm = legendre_p_assoc(l, ma as i32, theta.cos());
    match m.cmp(&0) {
        std::cmp::Ordering::Greater => {
            std::f64::consts::SQRT_2 * norm * plm * (ma as f64 * phi).cos()
        }
        std::cmp::Ordering::Less => {
            std::f64::consts::SQRT_2 * norm * plm * (ma as f64 * phi).sin()
        }
        std::cmp::Ordering::Equal => norm * plm,
    }
}

/// Nodes and weights of the n-point Gauss-Legendre rule on [−1, 1]
/// (NR `gauleg`: Newton iteration on Pₙ from the Chebyshev-like initial
/// guess). Integrates polynomials up to degree 2n−1 exactly.
///
/// # Panics
/// Panics if n = 0.
#[must_use]
pub fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n > 0, "gauss_legendre_nodes requires n > 0");
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess for the i-th root (descending).
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        loop {
            // Evaluate P_n and its derivative at z.
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 1e-15 {
                nodes[i] = -z;
                nodes[n - 1 - i] = z;
                let w = 2.0 / ((1.0 - z * z) * pp * pp);
                weights[i] = w;
                weights[n - 1 - i] = w;
                break;
            }
        }
    }
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_legendre_p_low_orders() {
        for &x in &[-0.9, -0.3, 0.0, 0.5, 1.0] {
            assert!(approx(legendre_p(0, x), 1.0, 1e-15));
            assert!(approx(legendre_p(1, x), x, 1e-15));
            assert!(approx(legendre_p(2, x), 0.5 * (3.0 * x * x - 1.0), 1e-14));
            assert!(approx(
                legendre_p(3, x),
                0.5 * (5.0 * x * x * x - 3.0 * x),
                1e-14
            ));
        }
        // P_n(1) = 1 for all n.
        for n in 0..12 {
            assert!(approx(legendre_p(n, 1.0), 1.0, 1e-12));
        }
    }

    #[test]
    fn test_assoc_legendre_known() {
        // P_1^1(x) = -sqrt(1-x^2) (Condon-Shortley); P_2^1(x) = -3x sqrt(1-x^2)
        for &x in &[-0.7_f64, 0.0, 0.4] {
            let s = (1.0 - x * x).sqrt();
            assert!(approx(legendre_p_assoc(1, 1, x), -s, 1e-14));
            assert!(approx(legendre_p_assoc(2, 1, x), -3.0 * x * s, 1e-13));
            assert!(approx(legendre_p_assoc(2, 2, x), 3.0 * (1.0 - x * x), 1e-13));
            // m = 0 reduces to P_l.
            assert!(approx(legendre_p_assoc(3, 0, x), legendre_p(3, x), 1e-13));
        }
    }

    #[test]
    fn test_assoc_legendre_negative_m() {
        // P_2^{-1} = -(1/6) P_2^1... specifically (-1)^1 (2-1)!/(2+1)! = -1/6.
        let x = 0.3;
        assert!(approx(
            legendre_p_assoc(2, -1, x),
            -legendre_p_assoc(2, 1, x) / 6.0,
            1e-14
        ));
    }

    #[test]
    fn test_spherical_harmonic_y00_y10() {
        // Y_0^0 = 1/sqrt(4 pi); Y_1^0 = sqrt(3/(4 pi)) cos(theta)
        let y00 = spherical_harmonic_real(0, 0, 0.7, 1.3);
        assert!(approx(y00, 0.5 / PI.sqrt(), 1e-14));
        let theta = 0.9;
        let y10 = spherical_harmonic_real(1, 0, theta, 2.0);
        assert!(approx(y10, (3.0 / (4.0 * PI)).sqrt() * theta.cos(), 1e-13));
    }

    #[test]
    fn test_gauss_legendre_five_matches_hardcoded() {
        let (nodes, weights) = gauss_legendre_nodes(5);
        let expected_nodes = [
            -0.906_179_845_938_664,
            -0.538_469_310_105_683,
            0.0,
            0.538_469_310_105_683,
            0.906_179_845_938_664,
        ];
        let expected_weights = [
            0.236_926_885_056_189_1,
            0.478_628_670_499_366_5,
            0.568_888_888_888_888_9,
            0.478_628_670_499_366_5,
            0.236_926_885_056_189_1,
        ];
        for i in 0..5 {
            assert!(approx(nodes[i], expected_nodes[i], 1e-12), "node {i}");
            assert!(approx(weights[i], expected_weights[i], 1e-12), "weight {i}");
        }
    }

    #[test]
    fn test_gauss_legendre_exactness() {
        // n-point rule integrates x^(2n-1) and below exactly on [-1, 1].
        let (nodes, weights) = gauss_legendre_nodes(8);
        // ∫ x^6 dx over [-1,1] = 2/7.
        let approx_int: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(&x, &w)| w * x.powi(6))
            .sum();
        assert!(approx(approx_int, 2.0 / 7.0, 1e-13));
        // Weights sum to 2.
        assert!(approx(weights.iter().sum::<f64>(), 2.0, 1e-13));
    }
}
