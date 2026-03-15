// ---------------------------------------------------------------------------
// Shannon Entropy
// ---------------------------------------------------------------------------

/// H = -Σ pi × log₂(pi), skipping pi = 0.
pub fn shannon_entropy(probabilities: &[f64]) -> f64 {
    assert!(!probabilities.is_empty(), "probabilities must be non-empty");
    -probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum::<f64>()
}

/// H = -Σ pi × ln(pi), in nats.
pub fn shannon_entropy_nats(probabilities: &[f64]) -> f64 {
    assert!(!probabilities.is_empty(), "probabilities must be non-empty");
    -probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f64>()
}

/// H_max = log₂(N) for N equally likely symbols.
pub fn max_entropy(n_symbols: usize) -> f64 {
    assert!(n_symbols > 0, "n_symbols must be positive");
    (n_symbols as f64).log2()
}

/// Identity function returning the conditional entropy value. Provided for API
/// completeness so callers can be explicit about what the quantity represents.
pub fn entropy_rate(conditional_entropy: f64) -> f64 {
    conditional_entropy
}

// ---------------------------------------------------------------------------
// Relative Entropy & Divergence
// ---------------------------------------------------------------------------

/// D_KL(P||Q) = Σ pi × ln(pi/qi), skipping pi = 0.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| {
            assert!(qi > 0.0, "q must be nonzero where p is nonzero");
            pi * (pi / qi).ln()
        })
        .sum()
}

/// Jensen-Shannon divergence: JSD(P||Q) = (D_KL(P||M) + D_KL(Q||M)) / 2
/// where M = (P + Q) / 2.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (kl_divergence(p, &m) + kl_divergence(q, &m)) / 2.0
}

/// H(P, Q) = -Σ pi × log₂(qi).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");
    -p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| {
            assert!(qi > 0.0, "q must be nonzero where p is nonzero");
            pi * qi.log2()
        })
        .sum::<f64>()
}

// ---------------------------------------------------------------------------
// Mutual Information
// ---------------------------------------------------------------------------

/// I(X;Y) = Σ p(x,y) × ln(p(x,y) / (p(x) × p(y))).
/// `joint` is an nx×ny row-major probability table.
pub fn mutual_information(
    joint: &[f64],
    marginal_x: &[f64],
    marginal_y: &[f64],
    nx: usize,
    ny: usize,
) -> f64 {
    assert_eq!(joint.len(), nx * ny, "joint must have nx*ny elements");
    assert_eq!(marginal_x.len(), nx, "marginal_x must have nx elements");
    assert_eq!(marginal_y.len(), ny, "marginal_y must have ny elements");

    let mut mi = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            let pxy = joint[i * ny + j];
            if pxy > 0.0 {
                let px = marginal_x[i];
                let py = marginal_y[j];
                assert!(px > 0.0 && py > 0.0, "marginals must be nonzero where joint is nonzero");
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }
    mi
}

/// H(Y|X) = -Σ p(x,y) × ln(p(y|x)).
/// `joint` is an nx×ny row-major probability table, `marginal_condition` are
/// the marginal probabilities p(x).
pub fn conditional_entropy(
    joint: &[f64],
    marginal_condition: &[f64],
    nx: usize,
    ny: usize,
) -> f64 {
    assert_eq!(joint.len(), nx * ny, "joint must have nx*ny elements");
    assert_eq!(marginal_condition.len(), nx, "marginal_condition must have nx elements");

    let mut h = 0.0;
    for i in 0..nx {
        let px = marginal_condition[i];
        if px <= 0.0 {
            continue;
        }
        for j in 0..ny {
            let pxy = joint[i * ny + j];
            if pxy > 0.0 {
                let p_y_given_x = pxy / px;
                h -= pxy * p_y_given_x.ln();
            }
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Channel Capacity
// ---------------------------------------------------------------------------

/// H(p) = -p × log₂(p) - (1-p) × log₂(1-p).
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// C = 1 - H(p) for a binary symmetric channel with crossover probability p.
pub fn binary_symmetric_channel_capacity(error_prob: f64) -> f64 {
    1.0 - binary_entropy(error_prob)
}

// ---------------------------------------------------------------------------
// Data Compression Bounds
// ---------------------------------------------------------------------------

/// R = original_bits / compressed_bits.
pub fn compression_ratio(original_bits: f64, compressed_bits: f64) -> f64 {
    assert!(compressed_bits > 0.0, "compressed_bits must be positive");
    original_bits / compressed_bits
}

/// D = 1 - H / H_max.
pub fn redundancy(entropy: f64, max_entropy: f64) -> f64 {
    assert!(max_entropy > 0.0, "max_entropy must be positive");
    1.0 - entropy / max_entropy
}

/// η = H / L where L is average code length.
pub fn efficiency(entropy: f64, avg_code_length: f64) -> f64 {
    assert!(avg_code_length > 0.0, "avg_code_length must be positive");
    entropy / avg_code_length
}

// ---------------------------------------------------------------------------
// Fisher Information
// ---------------------------------------------------------------------------

/// I(μ) = 1/σ² for a Gaussian when estimating the mean.
pub fn fisher_information_gaussian(sigma: f64) -> f64 {
    assert!(sigma > 0.0, "sigma must be positive");
    1.0 / (sigma * sigma)
}

/// Cramér-Rao lower bound: var(θ̂) ≥ 1 / I(θ).
pub fn cramer_rao_bound(fisher_info: f64) -> f64 {
    assert!(fisher_info > 0.0, "fisher_info must be positive");
    1.0 / fisher_info
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    #[test]
    fn test_fair_coin_entropy() {
        let probs = [0.5, 0.5];
        assert!(approx(shannon_entropy(&probs), 1.0));
    }

    #[test]
    fn test_fair_die_entropy() {
        let probs = [1.0 / 6.0; 6];
        let expected = (6.0_f64).log2();
        assert!(approx(shannon_entropy(&probs), expected));
    }

    #[test]
    fn test_max_entropy() {
        assert!(approx(max_entropy(6), (6.0_f64).log2()));
        assert!(approx(max_entropy(2), 1.0));
    }

    #[test]
    fn test_shannon_entropy_nats_fair_coin() {
        let probs = [0.5, 0.5];
        let expected = (2.0_f64).ln();
        assert!(approx(shannon_entropy_nats(&probs), expected));
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = [0.25, 0.25, 0.25, 0.25];
        assert!(approx(kl_divergence(&p, &p), 0.0));
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        let jsd_pq = js_divergence(&p, &q);
        let jsd_qp = js_divergence(&q, &p);
        assert!(approx(jsd_pq, jsd_qp));
    }

    #[test]
    fn test_js_divergence_nonnegative() {
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        assert!(js_divergence(&p, &q) >= 0.0);
    }

    #[test]
    fn test_binary_channel_capacity_zero_error() {
        assert!(approx(binary_symmetric_channel_capacity(0.0), 1.0));
    }

    #[test]
    fn test_binary_channel_capacity_half_error() {
        assert!(approx(binary_symmetric_channel_capacity(0.5), 0.0));
    }

    #[test]
    fn test_binary_entropy_extremes() {
        assert!(approx(binary_entropy(0.0), 0.0));
        assert!(approx(binary_entropy(1.0), 0.0));
        assert!(approx(binary_entropy(0.5), 1.0));
    }

    #[test]
    fn test_mutual_information_independent() {
        // Two independent uniform binary variables
        let joint = [0.25, 0.25, 0.25, 0.25];
        let mx = [0.5, 0.5];
        let my = [0.5, 0.5];
        assert!(approx(mutual_information(&joint, &mx, &my, 2, 2), 0.0));
    }

    #[test]
    fn test_cross_entropy_geq_entropy() {
        let p = [0.7, 0.2, 0.1];
        let q = [0.5, 0.3, 0.2];
        let h = shannon_entropy(&p);
        let hpq = cross_entropy(&p, &q);
        assert!(hpq >= h - TOLERANCE);
    }

    #[test]
    fn test_cross_entropy_equals_entropy_for_same_distribution() {
        let p = [0.5, 0.3, 0.2];
        let h = shannon_entropy(&p);
        let hpq = cross_entropy(&p, &p);
        assert!(approx(h, hpq));
    }

    #[test]
    fn test_conditional_entropy() {
        // Perfectly correlated: H(Y|X) = 0
        let joint = [0.5, 0.0, 0.0, 0.5];
        let mx = [0.5, 0.5];
        assert!(approx(conditional_entropy(&joint, &mx, 2, 2), 0.0));
    }

    #[test]
    fn test_compression_ratio() {
        assert!(approx(compression_ratio(100.0, 50.0), 2.0));
    }

    #[test]
    fn test_redundancy() {
        assert!(approx(redundancy(1.0, 2.0), 0.5));
        assert!(approx(redundancy(2.0, 2.0), 0.0));
    }

    #[test]
    fn test_efficiency() {
        assert!(approx(efficiency(2.0, 2.5), 0.8));
    }

    #[test]
    fn test_fisher_information_gaussian() {
        assert!(approx(fisher_information_gaussian(2.0), 0.25));
        assert!(approx(fisher_information_gaussian(1.0), 1.0));
    }

    #[test]
    fn test_cramer_rao_bound() {
        assert!(approx(cramer_rao_bound(4.0), 0.25));
    }

    #[test]
    fn test_entropy_rate_identity() {
        assert!(approx(entropy_rate(1.234), 1.234));
    }
}
