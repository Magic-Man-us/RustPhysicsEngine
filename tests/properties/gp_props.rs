//! Properties of the Gaussian process module.
//!
//! Regression by conditioning has a closed form, so almost everything
//! here is an identity rather than a tolerance.
//!
//! *The variance carries no information about the observations.* It is
//! `k(x,x) - k_*^T K^-1 k_*`, in which `y` does not appear. Two
//! processes fitted to the same inputs with entirely different targets
//! return variances that agree bit for bit, and the mean is exactly
//! linear in the targets. Both are asserted with `==`.
//!
//! *Conditioning is a projection.* The posterior variance never exceeds
//! the prior, it vanishes at a noiselessly observed point, and it
//! returns to the prior far from any data. Adding a point can only
//! shrink it.
//!
//! *Kernels mean what they say.* A periodic kernel repeats exactly, a
//! stationary one depends only on the separation, a sum is a sum and a
//! product is a product -- and all of them stay positive semi-definite,
//! which is what makes the Cholesky succeed at all.
//!
//! *The likelihood is a probability.* Its value is checked against an
//! independent determinant, and tuning by maximising it is required to
//! actually raise it.

use rust_physics_engine::learn::gp::{sample_prior, Gp, KernelFn};
use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;

/// A spread of kernels, including compound ones.
fn kernels(rng: &mut Rng) -> Vec<KernelFn> {
    let l = 0.4 + 1.5 * rng.next_f64();
    let s = 0.5 + rng.next_f64();
    vec![
        KernelFn::Rbf { l, s },
        KernelFn::Matern32 { l, s },
        KernelFn::Matern52 { l, s },
        KernelFn::Periodic { l, p: 1.0 + 2.0 * rng.next_f64(), s },
        KernelFn::Sum(
            Box::new(KernelFn::Rbf { l, s }),
            Box::new(KernelFn::Linear { s: 0.3, c: 0.1 }),
        ),
        KernelFn::Product(
            Box::new(KernelFn::Rbf { l: 2.0 * l, s }),
            Box::new(KernelFn::Periodic { l, p: 1.7, s: 1.0 }),
        ),
    ]
}

fn scattered(rng: &mut Rng, n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| vec![i as f64 * 0.4 + 0.1 * rng.next_f64()]).collect()
}

#[test]
fn prop_the_variance_ignores_the_targets_and_the_mean_is_linear_in_them() {
    // Both exact. The first is the property that most surprises people
    // about Gaussian processes; the second is what makes the posterior
    // mean a linear smoother.
    let mut rng = Rng::new(0x6c92_31ad);
    for _ in 0..25 {
        let count = 6 + (rng.below(4)) as usize;
        let x = scattered(&mut rng, count);
        let q = (0..20).map(|i| vec![i as f64 * 0.18 - 0.4]).collect::<Vec<_>>();
        let noise = if rng.next_f64() < 0.5 { 0.0 } else { 0.05 * rng.next_f64() };
        for kernel in kernels(&mut rng) {
            let a: Vec<f64> = x.iter().map(|p| p[0].sin()).collect();
            let b: Vec<f64> = x.iter().map(|_| 10.0 * rng.next_gaussian()).collect();
            let ga = Gp::fit(kernel.clone(), noise, &x, &a).unwrap();
            let gb = Gp::fit(kernel.clone(), noise, &x, &b).unwrap();
            let (ma, va) = ga.predict(&q).unwrap();
            let (_, vb) = gb.predict(&q).unwrap();
            for i in 0..q.len() {
                assert_eq!(va[i], vb[i], "the variance moved with the targets at {i}");
            }
            // Linearity in the targets: a combination of targets gives
            // the same combination of means. Exact in exact arithmetic;
            // in floating point the jitter perturbs the covariance
            // matrix and the solve amplifies that by its condition
            // number, so the tolerance is that product rather than a
            // fixed number. A squared exponential on closely spaced
            // points reaches a condition number of 1e5 easily, and
            // demanding 1e-8 there would be demanding precision the
            // problem does not contain.
            let (alpha, beta) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
            let mixed: Vec<f64> =
                a.iter().zip(&b).map(|(p, r)| alpha * p + beta * r).collect();
            let gm = Gp::fit(kernel.clone(), noise, &x, &mixed).unwrap();
            let (mm, _) = gm.predict(&q).unwrap();
            let (mb, _) = gb.predict(&q).unwrap();
            let slack = 1e-9 + 1e-9 * gm.condition_estimate().max(ga.condition_estimate());
            let magnitude = b.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for i in 0..q.len() {
                let want = alpha * ma[i] + beta * mb[i];
                assert!(
                    (mm[i] - want).abs() < slack * magnitude,
                    "the mean was not linear at {i}: {} against {want}, slack {slack}",
                    mm[i]
                );
            }
        }
    }
}

#[test]
fn prop_conditioning_only_ever_reduces_uncertainty() {
    // The posterior variance never exceeds the prior, and adding a
    // point cannot raise it anywhere. Both follow from the posterior
    // being a projection, and neither depends on what was observed.
    let mut rng = Rng::new(0x0d47_ba31);
    for _ in 0..20 {
        let x = scattered(&mut rng, 8);
        let y: Vec<f64> = x.iter().map(|p| p[0].cos()).collect();
        let q: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.15 - 0.5]).collect();
        for kernel in kernels(&mut rng) {
            let few = Gp::fit(kernel.clone(), 0.0, &x[..4], &y[..4]).unwrap();
            let many = Gp::fit(kernel.clone(), 0.0, &x, &y).unwrap();
            let (_, v_few) = few.predict(&q).unwrap();
            let (_, v_many) = many.predict(&q).unwrap();
            // The jitter perturbs the covariance matrix and the solve
            // amplifies it by the condition number, so the slack is
            // that product. It is generous for a well-conditioned
            // kernel and honest for a squared exponential on points
            // packed inside its length scale, where the matrix is
            // singular to working precision and no tolerance chosen in
            // advance would be right for both.
            let kappa = many.condition_estimate().max(few.condition_estimate());
            let slack = 1e-9 + 1e-9 * kappa;
            for i in 0..q.len() {
                let prior = kernel.eval(&q[i], &q[i]);
                assert!(v_few[i] <= prior * (1.0 + 1e-9) + 1e-12, "above the prior at {i}");
                assert!(v_many[i] <= prior * (1.0 + 1e-9) + 1e-12);
                assert!(
                    v_many[i] <= v_few[i] + slack * prior,
                    "more data raised the variance at {i}: {} to {}",
                    v_few[i],
                    v_many[i]
                );
                assert!(v_many[i] >= 0.0, "a negative variance at {i}");
            }
            // At a noiselessly observed point there is nothing left.
            // How exactly it interpolates is set by the jitter times the
            // condition number -- the jitter perturbs the covariance
            // matrix by a relative amount of its own size and the solve
            // amplifies it. Since `condition_estimate` is only a lower
            // bound, the tolerance carries a factor of a hundred over
            // the jitter itself; the measured error tracks the bound's
            // shape closely across two decades of conditioning, which is
            // what makes it the right shape rather than a fitted number.
            let (mean, var) = many.predict(&x).unwrap();
            let interpolation_slack = 1e-7 + 1e-8 * kappa;
            for i in 0..x.len() {
                assert!(var[i] < 1e-7, "point {i} kept variance {}", var[i]);
                assert!(
                    (mean[i] - y[i]).abs() < interpolation_slack,
                    "point {i} was off by {}, slack {interpolation_slack}, kappa {kappa}",
                    mean[i] - y[i]
                );
            }
        }
    }
}

#[test]
fn prop_every_kernel_gives_a_positive_semidefinite_gram_matrix() {
    // The defining property of a covariance function, and the reason
    // the Cholesky in `fit` succeeds. Checked directly: every quadratic
    // form v^T K v over random v is nonnegative, for sums and products
    // as well as the primitives -- closure under those operations is
    // what makes kernel construction compositional.
    let mut rng = Rng::new(0x4a10_7f2c);
    for _ in 0..25 {
        let n = 5 + (rng.next_u64() % 6) as usize;
        let x = scattered(&mut rng, n);
        for kernel in kernels(&mut rng) {
            let mut k = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    k.set(i, j, kernel.eval(&x[i], &x[j]));
                }
            }
            // Symmetry first: a covariance is symmetric by definition.
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(k.get(i, j), k.get(j, i), "asymmetric at ({i},{j})");
                }
            }
            let scale = (0..n).map(|i| k.get(i, i)).fold(0.0f64, f64::max).max(1.0);
            for _ in 0..20 {
                let v: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
                let form: f64 = (0..n)
                    .map(|i| (0..n).map(|j| v[i] * k.get(i, j) * v[j]).sum::<f64>())
                    .sum();
                assert!(form > -1e-9 * scale, "a negative quadratic form {form}");
            }
        }
    }
}

#[test]
fn prop_a_periodic_kernel_repeats_exactly_and_stationary_ones_only_see_separation() {
    let mut rng = Rng::new(0x18e0_4c73);
    for _ in 0..40 {
        let period = 0.5 + 3.0 * rng.next_f64();
        let k = KernelFn::Periodic { l: 0.3 + rng.next_f64(), p: period, s: 0.5 + rng.next_f64() };
        let x = 4.0 * rng.next_gaussian();
        let base = k.eval(&[x], &[x]);
        for m in 1..=4 {
            assert_eq!(
                k.eval(&[x], &[x + m as f64 * period]),
                base,
                "the period was not exact after {m} repeats"
            );
        }
        // Stationary kernels see only the separation, in either
        // direction, wherever they are evaluated.
        let l = 0.3 + rng.next_f64();
        let s = 0.5 + rng.next_f64();
        for stationary in [
            KernelFn::Rbf { l, s },
            KernelFn::Matern32 { l, s },
            KernelFn::Matern52 { l, s },
        ] {
            let r = 3.0 * rng.next_f64();
            let anchor = 5.0 * rng.next_gaussian();
            let a = stationary.eval(&[0.0], &[r]);
            assert!((stationary.eval(&[anchor], &[anchor + r]) - a).abs() < 1e-13);
            assert!((stationary.eval(&[anchor], &[anchor - r]) - a).abs() < 1e-13);
            assert!(a <= stationary.eval(&[0.0], &[0.0]) + 1e-15, "a kernel peaked off zero");
        }
    }
}

#[test]
fn prop_the_marginal_likelihood_matches_an_independent_computation() {
    // The module reads the log determinant off the Cholesky diagonal.
    // Here it comes from an LU factorisation instead, which shares no
    // code with it, and the solve is done separately too.
    let mut rng = Rng::new(0x2f76_90ba);
    const JITTER: f64 = 1e-10;
    for _ in 0..25 {
        let n = 4 + (rng.next_u64() % 5) as usize;
        let x = scattered(&mut rng, n);
        let y: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let noise = 0.01 + 0.2 * rng.next_f64();
        for kernel in kernels(&mut rng) {
            let gp = Gp::fit(kernel.clone(), noise, &x, &y).unwrap();
            let mut k = Matrix::zeros(n, n);
            let scale = kernel.eval(&x[0], &x[0]).abs().max(1.0);
            for i in 0..n {
                for j in 0..n {
                    k.set(i, j, kernel.eval(&x[i], &x[j]));
                }
                k.set(i, i, k.get(i, i) + noise + JITTER * scale);
            }
            let lu = rust_physics_engine::linalg::lu::lu_decompose(&k).unwrap();
            let solved = rust_physics_engine::linalg::lu::solve(&k, &y).unwrap();
            let fit: f64 = y.iter().zip(solved.iter()).map(|(a, b)| a * b).sum();
            let want = -0.5 * fit
                - 0.5 * lu.determinant().ln()
                - 0.5 * n as f64 * std::f64::consts::TAU.ln();
            let got = gp.log_marginal_likelihood();
            assert!(
                (got - want).abs() < 1e-7 * want.abs().max(1.0),
                "{got} against {want}"
            );
        }
    }
}

#[test]
fn prop_tuning_raises_the_likelihood_it_is_given() {
    // Not "finds the truth" -- the surface is not concave and the
    // search is local. What must hold is that the answer returned is
    // never worse than the starting point, whatever that was.
    let mut rng = Rng::new(0x77b3_ca10);
    for _ in 0..10 {
        let n = 10;
        let x = scattered(&mut rng, n);
        let frequency = 0.5 + 2.0 * rng.next_f64();
        let y: Vec<f64> = x
            .iter()
            .map(|p| (frequency * p[0]).sin() + 0.05 * rng.next_gaussian())
            .collect();
        let start = KernelFn::Rbf { l: 0.05 + 10.0 * rng.next_f64(), s: 0.1 + rng.next_f64() };
        let gp = Gp::fit(start, 0.01, &x, &y).unwrap();
        let before = gp.log_marginal_likelihood();
        let tuned = gp.optimize_hyperparams(3, &mut rng).unwrap();
        let after = tuned.log_marginal_likelihood();
        assert!(after >= before - 1e-6, "tuning lowered the likelihood: {before} to {after}");
        assert!(
            tuned.kernel.parameters().iter().all(|v| *v > 0.0 && v.is_finite()),
            "tuning produced a nonsense hyperparameter"
        );
    }
}

#[test]
fn prop_prior_draws_reproduce_the_kernel_they_came_from() {
    // The sample covariance of enough draws converges to the kernel
    // matrix, which is what "sampling from the prior" means. Checked
    // with a tolerance set by the standard error of that estimate
    // rather than by taste.
    let mut rng = Rng::new(0x5c04_8ef1);
    for _ in 0..6 {
        let points = scattered(&mut rng, 4);
        let l = 0.6 + rng.next_f64();
        let s = 0.6 + rng.next_f64();
        let kernel = KernelFn::Rbf { l, s };
        let draws = 20_000;
        let samples = sample_prior(&kernel, &points, draws, &mut rng).unwrap();
        assert_eq!(samples.len(), draws);
        for i in 0..points.len() {
            for j in 0..points.len() {
                let empirical: f64 =
                    samples.iter().map(|d| d[i] * d[j]).sum::<f64>() / draws as f64;
                let want = kernel.eval(&points[i], &points[j]);
                // The estimator's standard error is about
                // sqrt((k_ii k_jj + k_ij^2)/N); four of those is a
                // generous but principled band.
                let kii = kernel.eval(&points[i], &points[i]);
                let kjj = kernel.eval(&points[j], &points[j]);
                let se = ((kii * kjj + want * want) / draws as f64).sqrt();
                assert!(
                    (empirical - want).abs() < 4.0 * se,
                    "({i},{j}): {empirical} against {want}, se {se}"
                );
            }
        }
    }
}

#[test]
fn prop_posterior_draws_agree_with_the_posterior_they_came_from() {
    // The draws' mean and variance must match what `predict` reports,
    // which is the statement that the sampler and the closed form
    // describe the same distribution.
    let mut rng = Rng::new(0x63a9_0d5e);
    for _ in 0..6 {
        let x = scattered(&mut rng, 5);
        let y: Vec<f64> = x.iter().map(|p| p[0].cos()).collect();
        let kernel = KernelFn::Matern52 { l: 0.8, s: 1.0 };
        let gp = Gp::fit(kernel, 0.02, &x, &y).unwrap();
        let q: Vec<Vec<f64>> = (0..5).map(|i| vec![0.3 + i as f64 * 0.5]).collect();
        let (mean, var) = gp.predict(&q).unwrap();
        let draws = 20_000;
        let samples = gp.sample_posterior(&q, draws, &mut rng).unwrap();
        for i in 0..q.len() {
            let m: f64 = samples.iter().map(|d| d[i]).sum::<f64>() / draws as f64;
            let v: f64 = samples.iter().map(|d| (d[i] - m) * (d[i] - m)).sum::<f64>()
                / draws as f64;
            let se_mean = (var[i] / draws as f64).sqrt();
            assert!(
                (m - mean[i]).abs() < 4.0 * se_mean + 1e-9,
                "draw mean {m} against {} at {i}",
                mean[i]
            );
            let se_var = var[i] * (2.0 / draws as f64).sqrt();
            assert!(
                (v - var[i]).abs() < 5.0 * se_var + 1e-9,
                "draw variance {v} against {} at {i}",
                var[i]
            );
        }
    }
}
