//! Properties tying `stochastic::rmt` and `stochastic::extreme` to the rest
//! of the crate and to each other.
//!
//! The individual modules check each formula against its definition. These
//! check the results that connect a formula to something derived entirely
//! independently of it: that the moments of Wigner's semicircle are the
//! Catalan numbers the combinatorics module counts, that the eigenvalues a
//! linear algebra routine returns for a random covariance matrix land inside
//! the band a closed-form density predicts, and that the two separate routes
//! into a distribution's tail agree on its shape.

use rust_physics_engine::discrete::combinatorics::catalan;
use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::stochastic::extreme::{
    block_maxima, copula_clayton, copula_fit_tau, copula_frank, copula_gaussian_sample,
    copula_gumbel, copula_tau, empirical_copula, extremal_index, gev_cdf, gev_fit, gev_quantile,
    gpd_fit, kendall_tau, return_level, return_period, spearman_rho, CopulaFamily,
};
use rust_physics_engine::stochastic::rmt::{
    correlation_matrix_denoise_mp, goe_sample, level_spacing_ratio, mp_edges, symmetric_spectrum,
    wigner_semicircle, wishart_sample,
};

/// A value in `0..n` from the high bits: `% n` reads the low bits of the
/// linear congruential generator, where bit `b` has period `2^(b+1)`.
fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.next_f64()
}

#[test]
fn prop_the_semicircle_moments_are_the_catalan_numbers() {
    // The deepest identity in the module, and the reason random matrix theory
    // has a combinatorial side at all. The 2k-th moment of the semicircle of
    // radius 2 counts the ways to pair 2k points on a circle without
    // crossings -- the k-th Catalan number -- because that is what survives
    // when the expectation of a trace of a matrix power is expanded and every
    // non-planar pairing is suppressed by a power of 1/n.
    //
    // The two sides are computed by routes that share nothing: a numerical
    // integral of a density here, an exact integer recurrence in
    // `discrete::combinatorics` there.
    let steps = 400_000usize;
    for k in 0..=8u64 {
        // Substituting x = 2 sin(theta) flattens the square-root edges, where
        // a uniform grid converges at only h^(1/2).
        let h = std::f64::consts::PI / steps as f64;
        let moment: f64 = (0..steps)
            .map(|i| {
                let theta = -std::f64::consts::FRAC_PI_2 + (i as f64 + 0.5) * h;
                let x = 2.0 * theta.sin();
                x.powi(2 * k as i32) * wigner_semicircle(x, 2.0) * 2.0 * theta.cos() * h
            })
            .sum();
        let expected = catalan(k).to_f64();
        assert!(
            (moment - expected).abs() < 1e-6 * (1.0 + expected),
            "moment {} of the semicircle is {moment}, not the Catalan number {expected}",
            2 * k
        );
    }

    // The odd moments vanish, since the density is even.
    for k in 0..4u64 {
        let h = std::f64::consts::PI / steps as f64;
        let odd: f64 = (0..steps)
            .map(|i| {
                let theta = -std::f64::consts::FRAC_PI_2 + (i as f64 + 0.5) * h;
                let x = 2.0 * theta.sin();
                x.powi(2 * k as i32 + 1) * wigner_semicircle(x, 2.0) * 2.0 * theta.cos() * h
            })
            .sum();
        assert!(odd.abs() < 1e-9, "odd moment {} came out {odd}", 2 * k + 1);
    }
}

#[test]
fn prop_a_noise_covariance_stays_inside_its_marchenko_pastur_band() {
    // Independent variables at any aspect ratio: a linear algebra routine
    // produces the eigenvalues, a closed-form density says where they may
    // fall, and neither knows about the other.
    let mut rng = Rng::new(0x0011_045E);
    for _ in 0..20 {
        let p = 20 + pick(&mut rng, 60);
        let n = p * (2 + pick(&mut rng, 6));
        let ratio = p as f64 / n as f64;
        let (lo, hi) = mp_edges(ratio, 1.0);

        let s = wishart_sample(n, p, &mut rng);
        let eigs = symmetric_spectrum(&s).unwrap();
        assert_eq!(eigs.len(), p);
        assert!(eigs.iter().all(|&v| v > 0.0), "a covariance eigenvalue was not positive");

        // Finite-sample fluctuation at the edges is of order p^(-2/3), so a
        // margin is needed; the band still has to bracket the spectrum.
        let margin = 0.35;
        assert!(
            eigs[0] > lo * (1.0 - margin) - 0.05,
            "ratio {ratio}: smallest eigenvalue {} below the band edge {lo}",
            eigs[0]
        );
        assert!(
            eigs[p - 1] < hi * (1.0 + margin),
            "ratio {ratio}: largest eigenvalue {} above the band edge {hi}",
            eigs[p - 1]
        );
        // The trace of a correlation-scaled covariance is p on average.
        let mean: f64 = eigs.iter().sum::<f64>() / p as f64;
        assert!((mean - 1.0).abs() < 0.12, "ratio {ratio}: mean eigenvalue {mean}");
        // A wider aspect ratio means a wider band, always.
        assert!(hi > lo);
    }
}

#[test]
fn prop_denoising_preserves_the_trace_and_never_widens_the_spectrum() {
    // Whatever the input, cleaning replaces a set of eigenvalues by their own
    // average. That cannot change their sum, and cannot spread them further
    // apart.
    let mut rng = Rng::new(0x00DE_0155);
    for _ in 0..20 {
        let p = 10 + pick(&mut rng, 30);
        let t = p * (2 + pick(&mut rng, 5));
        // A sample correlation matrix of independent columns.
        let data: Vec<Vec<f64>> =
            (0..t).map(|_| (0..p).map(|_| rng.next_gaussian()).collect()).collect();
        let means: Vec<f64> =
            (0..p).map(|j| data.iter().map(|r| r[j]).sum::<f64>() / t as f64).collect();
        let sds: Vec<f64> = (0..p)
            .map(|j| {
                (data.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>() / t as f64).sqrt()
            })
            .collect();
        let mut corr = Matrix::zeros(p, p);
        for a in 0..p {
            for b in a..p {
                let v: f64 = data
                    .iter()
                    .map(|r| (r[a] - means[a]) * (r[b] - means[b]))
                    .sum::<f64>()
                    / (t as f64 * sds[a] * sds[b]);
                corr.set(a, b, v);
                corr.set(b, a, v);
            }
        }

        let cleaned = correlation_matrix_denoise_mp(&corr, t as f64 / p as f64).unwrap();
        assert!(cleaned.is_symmetric(1e-9));
        let trace = |m: &Matrix| (0..p).map(|i| m.get(i, i)).sum::<f64>();
        assert!(
            (trace(&corr) - trace(&cleaned)).abs() < 1e-8 * p as f64,
            "the trace moved from {} to {}",
            trace(&corr),
            trace(&cleaned)
        );

        let before = symmetric_spectrum(&corr).unwrap();
        let after = symmetric_spectrum(&cleaned).unwrap();
        let spread = |v: &[f64]| v[v.len() - 1] - v[0];
        assert!(
            spread(&after) <= spread(&before) + 1e-8,
            "cleaning widened the spectrum: {} to {}",
            spread(&before),
            spread(&after)
        );
        // The eigenvalues are still non-negative, so the result is still a
        // valid correlation matrix.
        assert!(after[0] > -1e-8, "cleaning produced a negative eigenvalue {}", after[0]);
    }
}

#[test]
fn prop_a_random_spectrum_repels_more_than_random_points_do() {
    // The one statistic that needs no unfolding, over a range of sizes: a GOE
    // spectrum sits near 0.5307 and independent points near 2 ln 2 - 1.
    let mut rng = Rng::new(0x002A_710C);
    let poisson_value = 2.0 * 2.0f64.ln() - 1.0;
    for _ in 0..8 {
        let n = 60 + pick(&mut rng, 60);
        let spectrum = symmetric_spectrum(&goe_sample(n, &mut rng)).unwrap();
        let correlated = level_spacing_ratio(&spectrum);

        let mut points: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let uncorrelated = level_spacing_ratio(&points);

        assert!((0.0..=1.0).contains(&correlated));
        assert!((0.0..=1.0).contains(&uncorrelated));
        assert!(
            correlated > uncorrelated,
            "at n = {n} the spectrum ({correlated}) did not repel more than noise ({uncorrelated})"
        );
        assert!(
            (correlated - 0.5307).abs() < 0.09,
            "at n = {n} the spectrum gave {correlated}"
        );
        assert!(
            (uncorrelated - poisson_value).abs() < 0.09,
            "at n = {n} independent points gave {uncorrelated}"
        );
    }
}

#[test]
fn prop_the_two_routes_into_the_tail_agree_on_its_shape() {
    // Pickands-Balkema-de Haan across a range of tail indices: fitting a GEV
    // to block maxima and a generalised Pareto to threshold exceedances of the
    // same data must recover the same shape, though the two estimators share
    // no code and see different observations.
    let mut rng = Rng::new(0x7A11_0001);
    for step in 0..6 {
        let xi = 0.15 + step as f64 * 0.12;
        let raw: Vec<f64> =
            (0..40_000).map(|_| rng.next_f64().clamp(1e-12, 1.0 - 1e-12).powf(-xi)).collect();

        let maxima = block_maxima(&raw, 200);
        let (_, _, block_shape) = gev_fit(&maxima).unwrap();

        let mut sorted = raw.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold = sorted[raw.len() - 2000];
        let excesses: Vec<f64> =
            raw.iter().filter(|&&v| v > threshold).map(|&v| v - threshold).collect();
        let (_, pot_shape) = gpd_fit(&excesses).unwrap();

        assert!(
            (block_shape - xi).abs() < 0.15,
            "xi = {xi}: the block route gave {block_shape}"
        );
        assert!((pot_shape - xi).abs() < 0.10, "xi = {xi}: the threshold route gave {pot_shape}");
        assert!(
            (block_shape - pot_shape).abs() < 0.18,
            "xi = {xi}: the routes disagree, {block_shape} against {pot_shape}"
        );
    }
}

#[test]
fn prop_return_levels_and_periods_are_inverse_across_the_gev_family() {
    let mut rng = Rng::new(0x002E_7021);
    for _ in 0..200 {
        let mu = uniform(&mut rng, -20.0, 20.0);
        let sigma = uniform(&mut rng, 0.1, 10.0);
        let xi = uniform(&mut rng, -0.45, 0.45);
        let mut previous = f64::NEG_INFINITY;
        for &period in &[1.5f64, 2.0, 10.0, 100.0, 1000.0] {
            let level = return_level(mu, sigma, xi, period);
            assert!(level.is_finite(), "xi = {xi} gave a non-finite level");
            assert!(level > previous, "return levels are not increasing at period {period}");
            previous = level;

            let back = return_period(mu, sigma, xi, level);
            assert!(
                (back - period).abs() < 1e-8 * (1.0 + period),
                "period {period} came back as {back}"
            );
            // And the level really is the quantile it claims to be.
            assert!(
                (gev_cdf(level, mu, sigma, xi) - (1.0 - 1.0 / period)).abs() < 1e-12,
                "the level is not the 1 - 1/T quantile"
            );
        }
        // The quantile function is monotone across the whole range.
        let mut last = f64::NEG_INFINITY;
        for k in 1..40 {
            let q = gev_quantile(k as f64 / 40.0, mu, sigma, xi);
            assert!(q > last, "the quantile function is not increasing");
            last = q;
        }
    }
}

#[test]
fn prop_copula_parameters_survive_a_round_trip_through_kendall_tau() {
    // Sample from a copula at a known parameter, measure a rank statistic that
    // ignores the margins entirely, and invert the family's analytic relation.
    // The sampler and the tau formula are derived independently.
    let mut rng = Rng::new(0x00C0_7A00);
    for step in 0..5 {
        let cases: Vec<(CopulaFamily, f64, Vec<Vec<f64>>)> = vec![
            (
                CopulaFamily::Clayton,
                0.8 + step as f64 * 1.4,
                copula_clayton(0.8 + step as f64 * 1.4, 20_000, &mut rng),
            ),
            (
                CopulaFamily::Gumbel,
                1.3 + step as f64 * 0.8,
                copula_gumbel(1.3 + step as f64 * 0.8, 20_000, &mut rng),
            ),
            (
                CopulaFamily::Frank,
                1.0 + step as f64 * 2.5,
                copula_frank(1.0 + step as f64 * 2.5, 20_000, &mut rng),
            ),
        ];
        for (family, theta, data) in cases {
            let fitted = copula_fit_tau(&data, family).unwrap();
            assert!(
                (fitted - theta).abs() < 0.12 * (1.0 + theta),
                "{family:?} at {theta} came back as {fitted}"
            );

            // The measured tau must also match what the family's closed form
            // predicts for the true parameter.
            let x: Vec<f64> = data.iter().map(|r| r[0]).collect();
            let y: Vec<f64> = data.iter().map(|r| r[1]).collect();
            let measured = kendall_tau(&x, &y);
            assert!(
                (measured - copula_tau(family, theta)).abs() < 0.03,
                "{family:?} at {theta}: measured tau {measured} against {}",
                copula_tau(family, theta)
            );
            // Spearman's rho agrees on the sign and, for these positively
            // dependent families, reads larger.
            assert!(spearman_rho(&x, &y) > measured, "{family:?}: rho did not exceed tau");
        }
    }

    // The Gaussian family, whose parameter is a correlation.
    for step in 0..5 {
        let rho = -0.8 + step as f64 * 0.4;
        let corr = Matrix::from_rows(&[&[1.0, rho], &[rho, 1.0]]).unwrap();
        let data = copula_gaussian_sample(&corr, 20_000, &mut rng).unwrap();
        let fitted = copula_fit_tau(&data, CopulaFamily::Gaussian).unwrap();
        assert!((fitted - rho).abs() < 0.05, "Gaussian at {rho} came back as {fitted}");
    }
}

#[test]
fn prop_rank_statistics_ignore_the_margins_entirely() {
    // A copula sample transformed through wildly different marginal
    // distributions has to give exactly the same rank correlations and exactly
    // the same pseudo-observations. This is the separation of copula from
    // margins, stated as an identity rather than an approximation.
    let mut rng = Rng::new(0x00C0_2A11);
    for step in 0..12 {
        let data = match step % 3 {
            0 => copula_clayton(1.0 + step as f64 * 0.3, 800, &mut rng),
            1 => copula_gumbel(1.2 + step as f64 * 0.2, 800, &mut rng),
            _ => copula_frank(1.0 + step as f64 * 0.7, 800, &mut rng),
        };
        let u: Vec<f64> = data.iter().map(|r| r[0]).collect();
        let v: Vec<f64> = data.iter().map(|r| r[1]).collect();
        let (tau, rho) = (kendall_tau(&u, &v), spearman_rho(&u, &v));
        let pseudo = empirical_copula(&data).unwrap();

        // Any strictly increasing transform of either margin.
        let transformed: Vec<Vec<f64>> = data
            .iter()
            .map(|r| {
                vec![
                    // A normal-ish quantile via a monotone rational map, and a
                    // heavy-tailed Pareto transform on the other coordinate.
                    (r[0] / (1.0 - r[0]).max(1e-12)).ln(),
                    (1.0 - r[1]).max(1e-12).powf(-3.0) * 1e6,
                ]
            })
            .collect();
        let tu: Vec<f64> = transformed.iter().map(|r| r[0]).collect();
        let tv: Vec<f64> = transformed.iter().map(|r| r[1]).collect();

        assert!((kendall_tau(&tu, &tv) - tau).abs() < 1e-12, "tau moved under a transform");
        assert!((spearman_rho(&tu, &tv) - rho).abs() < 1e-9, "rho moved under a transform");
        let pseudo_t = empirical_copula(&transformed).unwrap();
        for (a, b) in pseudo.iter().zip(&pseudo_t) {
            assert!((a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
        }
    }
}

#[test]
fn prop_the_extremal_index_counts_clusters_not_exceedances() {
    // Independent exceedances give an index of one; running a maximum over a
    // window of m echoes each large value m times and drives the index to 1/m.
    // The number of exceedances barely changes -- only how many distinct
    // events they represent.
    let mut rng = Rng::new(0x00E2_C105);
    for m in 1..=5usize {
        let base: Vec<f64> = (0..60_000).map(|_| rng.next_gaussian()).collect();
        let series: Vec<f64> = if m == 1 {
            base.clone()
        } else {
            (m - 1..base.len())
                .map(|t| base[t + 1 - m..=t].iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .collect()
        };
        let index = extremal_index(&series, 2.2);
        assert!((0.0..=1.0).contains(&index), "the index left [0, 1]: {index}");
        assert!(
            (index - 1.0 / m as f64).abs() < 0.20,
            "a window of {m} gave an index of {index}, not {}",
            1.0 / m as f64
        );
    }

    // Monotone in the window length: more echoing means a lower index.
    let base: Vec<f64> = (0..60_000).map(|_| rng.next_gaussian()).collect();
    let mut previous = f64::INFINITY;
    for m in [1usize, 3, 6, 10] {
        let series: Vec<f64> = if m == 1 {
            base.clone()
        } else {
            (m - 1..base.len())
                .map(|t| base[t + 1 - m..=t].iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .collect()
        };
        let index = extremal_index(&series, 2.2);
        assert!(index < previous + 1e-9, "the index rose at a window of {m}");
        previous = index;
    }
}
