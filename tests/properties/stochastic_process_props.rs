//! Properties tying `stochastic::queueing` and `stochastic::timeseries` to
//! each other, to `stochastic::markov`, and to `transforms::fft`.
//!
//! Each module's own tests check it against its definitions. These check the
//! theorems that connect modules, which no single one of them can check
//! alone: Little's law across every queueing model at once, the two
//! independent routes from a birth-death chain to its stationary
//! distribution, the identity between a continuous-time chain and its
//! embedded discrete one, and -- on the time series side -- the equivalence
//! of the time-domain and frequency-domain descriptions of the same
//! second-order structure, checked against an FFT that knows nothing about
//! ARMA models.

use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::stochastic::queueing::{
    littles_law_check, mm1, mm1k, mm_inf, mmc, mmck, uniformization, Ctmc,
};
use rust_physics_engine::stochastic::timeseries::{acf, Arma};
use rust_physics_engine::transforms::fft::fft_any;

/// A value in `0..n` from the high bits: `% n` reads the low bits of the
/// linear congruential generator, where bit `b` has period `2^(b+1)`.
fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A uniform draw in `[lo, hi)`.
fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.next_f64()
}

/// The generator of a birth-death chain on `0..=k` with the given up and
/// down rates.
fn birth_death(lambda: f64, mu: f64, servers: usize, k: usize) -> Matrix {
    let n = k + 1;
    let mut q = Matrix::zeros(n, n);
    for i in 0..n {
        let up = if i + 1 < n { lambda } else { 0.0 };
        // With `servers` in parallel the service rate rises until they are
        // all busy and then stops.
        let down = if i > 0 { mu * i.min(servers) as f64 } else { 0.0 };
        if up > 0.0 {
            q.set(i, i + 1, up);
        }
        if down > 0.0 {
            q.set(i, i - 1, down);
        }
        q.set(i, i, -(up + down));
    }
    q
}

#[test]
fn prop_littles_law_holds_across_every_queueing_model() {
    // L = lambda W is a statement about areas under a sample path and assumes
    // nothing about the arrival or service distributions, so it has to hold
    // for every model in the module at every admissible parameter.
    let mut rng = Rng::new(0x_11771E);
    for _ in 0..300 {
        let mu = uniform(&mut rng, 0.2, 4.0);
        let c = 1 + pick(&mut rng, 6);
        let k = c + pick(&mut rng, 12);
        // Each unbounded model needs a rate its own server count can absorb;
        // a load that is comfortable for six servers saturates one. The
        // finite-capacity models are stable at any rate, since they block.
        let load = uniform(&mut rng, 0.05, 0.9);
        let one_server = load * mu;
        let many_servers = load * c as f64 * mu;

        let models = [
            mm1(one_server, mu),
            mmc(many_servers, mu, c),
            mm1k(uniform(&mut rng, 0.1, 5.0) * mu, mu, 1 + k),
            mmck(uniform(&mut rng, 0.1, 5.0) * c as f64 * mu, mu, c, k),
            mm_inf(many_servers, mu),
        ];
        for q in models {
            assert!(q.l.is_finite() && q.w.is_finite(), "a finite model reported infinities");
            let scale = 1.0 + q.l.abs();
            assert!(
                littles_law_check(q.l, q.lambda_eff, q.w).abs() < 1e-9 * scale,
                "L = {} against lambda W = {}",
                q.l,
                q.lambda_eff * q.w
            );
            assert!(
                littles_law_check(q.lq, q.lambda_eff, q.wq).abs() < 1e-9 * scale,
                "Lq = {} against lambda Wq = {}",
                q.lq,
                q.lambda_eff * q.wq
            );
            // The queue is a subset of the system, and the difference is
            // whoever is in service.
            assert!(q.lq <= q.l + 1e-9 && q.wq <= q.w + 1e-9);
            assert!(q.l >= 0.0 && q.lq >= -1e-12 && q.p0 >= 0.0);
        }
    }
}

#[test]
fn prop_the_product_form_and_the_balance_equations_agree() {
    // `mmck` builds its distribution from the birth-death product form, one
    // ratio at a time. `Ctmc::stationary` solves pi Q = 0 as a linear system
    // and knows nothing about queues. Two derivations, one answer.
    let mut rng = Rng::new(0x_B41A_11CE);
    for _ in 0..120 {
        let mu = uniform(&mut rng, 0.3, 3.0);
        let lambda = uniform(&mut rng, 0.2, 5.0);
        let c = 1 + pick(&mut rng, 4);
        let k = c + pick(&mut rng, 10);

        let chain = Ctmc::new(birth_death(lambda, mu, c, k)).unwrap();
        let pi = chain.stationary().unwrap();
        let analytic = mmck(lambda, mu, c, k);
        for n in 0..=k {
            assert!(
                (pi[n] - analytic.pn(n)).abs() < 1e-9,
                "state {n}: linear solve {} against product form {}",
                pi[n],
                analytic.pn(n)
            );
        }
        // And the mean built from that distribution is the one reported.
        let l: f64 = (0..=k).map(|n| n as f64 * pi[n]).sum();
        assert!((l - analytic.l).abs() < 1e-8, "mean {l} against {}", analytic.l);
    }
}

#[test]
fn prop_a_continuous_chain_is_its_jump_chain_weighted_by_holding_time() {
    // The bridge between `queueing::Ctmc` and `markov::MarkovChain`: a
    // continuous-time chain spends time in a state in proportion to how often
    // it visits times how long it stays, so pi is proportional to nu_i h_i
    // over the embedded chain's stationary law.
    let mut rng = Rng::new(0x_E3BE_DDED);
    for _ in 0..120 {
        let mu = uniform(&mut rng, 0.3, 3.0);
        let lambda = uniform(&mut rng, 0.3, 3.0);
        let c = 1 + pick(&mut rng, 3);
        let k = c + 1 + pick(&mut rng, 8);

        let chain = Ctmc::new(birth_death(lambda, mu, c, k)).unwrap();
        let pi = chain.stationary().unwrap();
        let jump = chain.embedded_chain().unwrap();
        let nu = jump.stationary();
        let h = chain.mean_holding_times();

        let weighted: Vec<f64> = nu.iter().zip(&h).map(|(&v, &t)| v * t).collect();
        let total: f64 = weighted.iter().sum();
        assert!(total.is_finite() && total > 0.0);
        for i in 0..chain.n() {
            assert!(
                (pi[i] - weighted[i] / total).abs() < 1e-8,
                "state {i}: {} against the reweighted jump chain {}",
                pi[i],
                weighted[i] / total
            );
        }
        // The jump chain must be a genuine stochastic matrix with no
        // self-transitions, since a continuous-time chain never jumps in place.
        for i in 0..chain.n() {
            let row: f64 = (0..chain.n()).map(|j| jump.p.get(i, j)).sum();
            assert!((row - 1.0).abs() < 1e-12);
            assert_eq!(jump.p.get(i, i), 0.0);
        }
    }
}

#[test]
fn prop_uniformization_is_a_distribution_that_relaxes_to_stationarity() {
    // Every partial sum of the Poisson mixture is a convex combination of
    // probability vectors, so the answer is a distribution at any horizon --
    // a property a truncated matrix exponential does not have. And as the
    // horizon grows it must approach the chain's stationary law, monotonically
    // in total variation.
    let mut rng = Rng::new(0x_0F1F_0417);
    for _ in 0..60 {
        let mu = uniform(&mut rng, 0.5, 3.0);
        let lambda = uniform(&mut rng, 0.5, 3.0);
        let c = 1 + pick(&mut rng, 3);
        let k = c + 1 + pick(&mut rng, 6);
        let chain = Ctmc::new(birth_death(lambda, mu, c, k)).unwrap();
        let pi = chain.stationary().unwrap();

        let n = chain.n();
        let mut start = vec![0.0; n];
        start[pick(&mut rng, n)] = 1.0;

        let mut previous = f64::INFINITY;
        for &t in &[0.25f64, 1.0, 4.0, 16.0, 64.0, 256.0] {
            let p = uniformization(&chain.q, &start, t, 1e-14).unwrap();
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "not a distribution at t = {t}");
            assert!(p.iter().all(|&v| v >= -1e-15), "a probability went negative at t = {t}");
            let distance: f64 =
                p.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
            assert!(
                distance <= previous + 1e-12,
                "the distance to stationary grew at t = {t}: {distance} after {previous}"
            );
            previous = distance;
        }
        assert!(previous < 1e-8, "still {previous} from stationary at t = 256");
    }
}

/// A random stationary autoregression of order `p`.
///
/// Sampling coefficients directly would almost always land outside the
/// stationary region for `p > 1`. Sampling *partial* autocorrelations in
/// `(-1, 1)` and running the Durbin-Levinson recursion forward is the
/// Barndorff-Nielsen-Schou map, which is a bijection onto exactly the
/// stationary region -- so every draw is stationary by construction.
fn random_stationary_ar(p: usize, rng: &mut Rng) -> Vec<f64> {
    let mut phi = vec![0.0f64; p + 1];
    let mut prev = vec![0.0f64; p + 1];
    for k in 1..=p {
        let kappa = uniform(rng, -0.85, 0.85);
        prev[..k].copy_from_slice(&phi[..k]);
        phi[k] = kappa;
        for j in 1..k {
            phi[j] = prev[j] - kappa * prev[k - j];
        }
    }
    phi[1..=p].to_vec()
}

#[test]
fn prop_the_spectral_density_integrates_to_the_impulse_response_variance() {
    // Parseval, in the form the time series module cares about: the integral
    // of the spectral density over [-pi, pi] is the process variance, and the
    // process variance is sigma^2 times the sum of squared psi weights. The
    // frequency-domain and time-domain descriptions of second-order structure
    // are the same object.
    let mut rng = Rng::new(0x_5EC7_2A11);
    let m = 20_000usize;
    let freqs: Vec<f64> = (0..m)
        .map(|i| {
            -std::f64::consts::PI
                + (i as f64 + 0.5) * 2.0 * std::f64::consts::PI / m as f64
        })
        .collect();

    for _ in 0..40 {
        let p = pick(&mut rng, 3);
        let q = pick(&mut rng, 3);
        if p == 0 && q == 0 {
            continue;
        }
        let ar = random_stationary_ar(p, &mut rng);
        let ma: Vec<f64> = (0..q).map(|_| uniform(&mut rng, -0.7, 0.7)).collect();
        let sigma2 = uniform(&mut rng, 0.3, 3.0);
        let model = Arma::new(ar, ma, sigma2, uniform(&mut rng, -5.0, 5.0));
        assert!(model.roots_check().0, "the sampler produced a non-stationary model");

        let dens = model.spectral_density(&freqs);
        assert!(dens.iter().all(|&v| v >= 0.0), "a spectral density went negative");
        let integral: f64 = dens.iter().sum::<f64>() * 2.0 * std::f64::consts::PI / m as f64;

        let psi = model.impulse_response(3000);
        let variance = model.sigma2 * psi.iter().map(|v| v * v).sum::<f64>();
        assert!(
            (integral - variance).abs() < 1e-4 * (1.0 + variance),
            "spectral integral {integral} against psi-weight variance {variance}"
        );
    }
}

#[test]
fn prop_the_averaged_periodogram_recovers_the_spectral_density() {
    // The strongest cross-module check available here: simulate an ARMA,
    // transform it with the crate's FFT, and compare the averaged periodogram
    // to the density the model computes from its own coefficients. Nothing in
    // `fft` knows about ARMA models and nothing in `spectral_density` knows
    // about the FFT, so agreement pins both.
    let mut rng = Rng::new(0x_7E12_0D06);
    for case in 0..6 {
        let ar = random_stationary_ar(1 + case % 2, &mut rng);
        let ma: Vec<f64> = if case % 3 == 0 { vec![] } else { vec![uniform(&mut rng, -0.6, 0.6)] };
        let model = Arma::new(ar, ma, 1.0, 0.0);

        let n = 512usize;
        let replicates = 200usize;
        let mut averaged = vec![0.0f64; n];
        for _ in 0..replicates {
            let x = model.simulate(n, &mut rng);
            let m = x.iter().sum::<f64>() / n as f64;
            let spectrum = fft_any(
                &x.iter()
                    .map(|&v| rust_physics_engine::fractals::Complex::new(v - m, 0.0))
                    .collect::<Vec<_>>(),
            );
            for k in 0..n {
                // I(w_k) = |sum_t x_t e^{-i w_k t}|^2 / (2 pi n), which is the
                // normalisation under which E[I] tends to f.
                averaged[k] += spectrum[k].norm_sq()
                    / (2.0 * std::f64::consts::PI * n as f64 * replicates as f64);
            }
        }

        // Compare over the interior Fourier frequencies: the zero frequency is
        // annihilated by centring the data, and the very lowest few carry the
        // worst of the periodogram's leakage bias.
        let freqs: Vec<f64> =
            (0..n).map(|k| 2.0 * std::f64::consts::PI * k as f64 / n as f64).collect();
        let truth = model.spectral_density(&freqs);
        let lo = 8usize;
        let hi = n / 2;
        let observed: f64 = averaged[lo..hi].iter().sum();
        let expected: f64 = truth[lo..hi].iter().sum();
        assert!(
            (observed - expected).abs() < 0.06 * expected,
            "case {case}: the averaged periodogram totalled {observed} against {expected}"
        );

        // And the shape, not merely the total: the peak of the density and of
        // the smoothed periodogram must fall in the same half of the band.
        let smooth = |v: &[f64], k: usize| -> f64 {
            let a = k.saturating_sub(6);
            let b = (k + 7).min(hi);
            v[a..b].iter().sum::<f64>() / (b - a) as f64
        };
        let argmax = |v: &[f64]| -> usize {
            (lo..hi).max_by(|&a, &b| smooth(v, a).partial_cmp(&smooth(v, b)).unwrap()).unwrap()
        };
        let (pa, pb) = (argmax(&averaged), argmax(&truth));
        assert!(
            pa.abs_diff(pb) < n / 8,
            "case {case}: the periodogram peaked at {pa} and the density at {pb}"
        );
    }
}

#[test]
fn prop_the_sample_autocorrelation_matches_the_model_it_came_from() {
    // The theoretical autocorrelation of an ARMA is gamma_h / gamma_0 with
    // gamma_h = sigma^2 sum_j psi_j psi_{j+h}. `acf` estimates the same
    // quantity from a realisation without ever seeing the coefficients.
    let mut rng = Rng::new(0x_ACF0_0007);
    for _ in 0..25 {
        let p = 1 + pick(&mut rng, 2);
        let q = pick(&mut rng, 2);
        // Keep the roots away from the unit circle: near it the process is
        // so persistent that a sample autocorrelation at any feasible length
        // carries a large downward bias, and the comparison would be testing
        // that bias rather than the identity.
        let ar: Vec<f64> = random_stationary_ar(p, &mut rng).iter().map(|v| v * 0.7).collect();
        let ma: Vec<f64> = (0..q).map(|_| uniform(&mut rng, -0.6, 0.6)).collect();
        let model = Arma::new(ar, ma, 1.0, 0.0);
        if !model.roots_check().0 {
            continue;
        }

        let psi = model.impulse_response(600);
        let gamma: Vec<f64> = (0..=6)
            .map(|h| psi.iter().zip(psi.iter().skip(h)).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let x = model.simulate(60_000, &mut rng);
        let sample = acf(&x, 6);
        for h in 1..=6 {
            let theoretical = gamma[h] / gamma[0];
            assert!(
                (sample[h] - theoretical).abs() < 0.05,
                "lag {h}: sample {} against theory {theoretical}",
                sample[h]
            );
        }
        assert_eq!(sample[0], 1.0);
    }
}
