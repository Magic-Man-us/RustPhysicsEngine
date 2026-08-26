//! Properties of the option pricing module.
//!
//! Derivative pricing is unusually well supplied with exact statements
//! that hold whatever the parameters, and they fall into three kinds.
//!
//! *Model-free identities* follow from the payoffs alone and would hold
//! for any arbitrage-free prices, whoever computed them: put-call parity,
//! the no-arbitrage bounds, homogeneity in the spot and strike together,
//! and the symmetry that exchanges the spot with the strike and the
//! interest rate with the dividend yield.
//!
//! *Degenerate cases* are where a general method must reproduce a special
//! one exactly: Merton with no jumps, Heston with no volatility of
//! volatility, and a barrier so far away it is never touched.
//!
//! *Convergence statements* say a numerical method approaches the closed
//! form at a stated rate. Those are the ones that catch a boundary
//! condition or a discretisation off by an order.

use rust_physics_engine::finance::options::{
    binomial_crr, black_scholes, bs_greeks, bs_pde_crank_nicolson, heston_price_mc,
    implied_volatility, longstaff_schwartz_american, merton_jump_price, monte_carlo_asian,
    monte_carlo_barrier, monte_carlo_european, monte_carlo_lookback, put_call_parity_check,
    svi_fit, trinomial, volatility_smile_svi, Barrier, Svi,
};
use rust_physics_engine::monte_carlo::Rng;

/// A randomised but sane option: spot, strike, maturity, rate, volatility,
/// dividend yield.
fn draw(rng: &mut Rng) -> (f64, f64, f64, f64, f64, f64) {
    let s = 20.0 + 200.0 * rng.next_f64();
    let k = s * (0.4 + 1.6 * rng.next_f64());
    let t = 0.02 + 5.0 * rng.next_f64();
    let r = -0.02 + 0.14 * rng.next_f64();
    let sigma = 0.03 + 0.9 * rng.next_f64();
    let q = 0.1 * rng.next_f64();
    (s, k, t, r, sigma, q)
}

#[test]
fn prop_put_call_parity_holds_for_every_parameter_set() {
    // Holding a call and selling a put is holding the forward. That is a
    // statement about payoffs, so no choice of parameters can break it and
    // a residue would be an error in the formula, not in the market.
    let mut rng = Rng::new(0x0F1A_5001);
    for _ in 0..600 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let call = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let put = black_scholes(s, k, t, r, sigma, q, false).unwrap();
        let residue = put_call_parity_check(call, put, s, k, t, r, q).abs();
        assert!(
            residue < 1e-10 * s.max(k),
            "S={s} K={k} T={t} r={r} vol={sigma} q={q} left {residue}"
        );
    }
}

#[test]
fn prop_prices_stay_inside_the_bounds_arbitrage_would_close() {
    let mut rng = Rng::new(0x0F1A_5002);
    for _ in 0..600 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let forward = s * (-q * t).exp();
        let strike = k * (-r * t).exp();
        let call = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let put = black_scholes(s, k, t, r, sigma, q, false).unwrap();
        let scale = 1e-10 * s.max(k);
        assert!(call >= (forward - strike).max(0.0) - scale, "the call fell under its floor");
        assert!(call <= forward + scale, "the call beat the share");
        assert!(put >= (strike - forward).max(0.0) - scale, "the put fell under its floor");
        assert!(put <= strike + scale, "the put beat the discounted strike");
        assert!(call.is_finite() && put.is_finite());
    }
}

#[test]
fn prop_a_price_is_homogeneous_in_the_spot_and_strike_together() {
    // Doubling the share price and the strike doubles the option: the
    // payoff scales and nothing else in the problem has units of money.
    // A formula that mixed up a level with a ratio would fail this.
    let mut rng = Rng::new(0x0F1A_5003);
    for _ in 0..300 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let factor = 0.1 + 20.0 * rng.next_f64();
        for call in [true, false] {
            let base = black_scholes(s, k, t, r, sigma, q, call).unwrap();
            let scaled = black_scholes(factor * s, factor * k, t, r, sigma, q, call).unwrap();
            assert!(
                (scaled - factor * base).abs() < 1e-9 * factor * base.max(1.0),
                "scaling by {factor} gave {scaled} not {}",
                factor * base
            );
        }
    }
}

#[test]
fn prop_a_call_is_a_put_with_the_spot_and_strike_exchanged() {
    // C(S, K, r, q) = P(K, S, q, r). Swapping the two assets swaps which
    // one is the numeraire, and the interest rate and the dividend yield
    // change places with them. It is exact and holds for every parameter.
    let mut rng = Rng::new(0x0F1A_5004);
    for _ in 0..400 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let call = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let mirrored = black_scholes(k, s, t, q, sigma, r, false).unwrap();
        assert!(
            (call - mirrored).abs() < 1e-10 * call.max(1.0),
            "S={s} K={k}: {call} against {mirrored}"
        );
    }
}

#[test]
fn prop_a_price_moves_the_way_its_greeks_say_it_does() {
    // Each Greek is checked against a central difference of the price it
    // differentiates, over randomised parameters rather than a handful of
    // chosen ones. A sign error or a missing carry term cannot survive.
    let mut rng = Rng::new(0x0F1A_5005);
    for _ in 0..300 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        if t < 0.05 || sigma < 0.05 {
            continue;
        }
        for call in [true, false] {
            let g = bs_greeks(s, k, t, r, sigma, q, call).unwrap();
            let price =
                |s: f64, t: f64, sigma: f64, r: f64| black_scholes(s, k, t, r, sigma, q, call).unwrap();
            let hs = 1e-5 * s;
            let delta = (price(s + hs, t, sigma, r) - price(s - hs, t, sigma, r)) / (2.0 * hs);
            assert!((g.delta - delta).abs() < 1e-5, "delta {} against {delta}", g.delta);
            // A second difference needs care from both sides: dividing by
            // h^2 amplifies round-off, so h cannot be small, and the
            // truncation error is O(h^2), so it cannot be large. Richardson
            // extrapolation over two steps removes the leading truncation
            // term and leaves a bound both effects fit under.
            let second = |h: f64| {
                (price(s + h, t, sigma, r) - 2.0 * price(s, t, sigma, r)
                    + price(s - h, t, sigma, r))
                    / (h * h)
            };
            let hg = 4e-3 * s;
            let gamma = (4.0 * second(0.5 * hg) - second(hg)) / 3.0;
            let noise = 8.0 * f64::EPSILON * price(s, t, sigma, r).max(s) / (0.25 * hg * hg);
            assert!(
                (g.gamma - gamma).abs() < 1e-4 * g.gamma.abs() + 20.0 * noise,
                "gamma {} against {gamma}",
                g.gamma
            );
            let h = 1e-5;
            let vega = (price(s, t, sigma + h, r) - price(s, t, sigma - h, r)) / (2.0 * h);
            assert!((g.vega - vega).abs() < 1e-4 * s, "vega {} against {vega}", g.vega);
            let rho = (price(s, t, sigma, r + h) - price(s, t, sigma, r - h)) / (2.0 * h);
            assert!((g.rho - rho).abs() < 1e-4 * s, "rho {} against {rho}", g.rho);
            let theta = -(price(s, t + h, sigma, r) - price(s, t - h, sigma, r)) / (2.0 * h);
            assert!((g.theta - theta).abs() < 1e-4 * s, "theta {} against {theta}", g.theta);
        }
        // Gamma and vega cannot tell a call from a put, because the
        // difference between them is a forward.
        let call = bs_greeks(s, k, t, r, sigma, q, true).unwrap();
        let put = bs_greeks(s, k, t, r, sigma, q, false).unwrap();
        assert!((call.gamma - put.gamma).abs() < 1e-14 * call.gamma.abs().max(1.0));
        assert!((call.vega - put.vega).abs() < 1e-12 * call.vega.abs().max(1.0));
        assert!((call.delta - put.delta - (-q * t).exp()).abs() < 1e-12);
    }
}

#[test]
fn prop_the_price_is_strictly_increasing_in_volatility() {
    // Which is why implied volatility is well defined at all: the map
    // from volatility to price is invertible wherever vega is non-zero.
    let mut rng = Rng::new(0x0F1A_5006);
    for _ in 0..200 {
        let (s, k, t, r, _, q) = draw(&mut rng);
        for call in [true, false] {
            let mut previous = f64::NEG_INFINITY;
            for step in 0..12 {
                let sigma = 0.02 + 0.15 * step as f64;
                let price = black_scholes(s, k, t, r, sigma, q, call).unwrap();
                assert!(price >= previous, "the price fell as volatility rose to {sigma}");
                previous = price;
            }
        }
    }
}

#[test]
fn prop_implied_volatility_inverts_the_price_wherever_vega_is_readable() {
    let mut rng = Rng::new(0x0F1A_5007);
    let mut inverted = 0usize;
    for _ in 0..400 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        for call in [true, false] {
            let price = black_scholes(s, k, t, r, sigma, q, call).unwrap();
            let vega = bs_greeks(s, k, t, r, sigma, q, call).unwrap().vega;
            let recovered = implied_volatility(price, s, k, t, r, q, call).unwrap();
            if vega < 1e-8 * price.max(1.0) {
                assert_eq!(recovered, None, "an unreadable price was answered anyway");
                continue;
            }
            let found = recovered.expect("a readable price has a volatility");
            assert!(
                (found - sigma).abs() < 1e-7 * sigma.max(1.0),
                "S={s} K={k} T={t}: recovered {found} not {sigma}"
            );
            inverted += 1;
        }
    }
    assert!(inverted > 500, "only {inverted} of the draws were readable at all");
}

#[test]
fn prop_the_binomial_lattice_is_arbitrage_free_at_every_step_count() {
    // Cox-Ross-Rubinstein picks its up-probability so that the discounted
    // price is a martingale *exactly*, so the tree is an arbitrage-free
    // market in its own right: it prices the linear payoff `C - P` to the
    // last bit whatever its step count, even seven steps, where the price
    // itself is nowhere near the continuous answer. That makes parity a
    // check on the lattice that does not depend on convergence at all.
    let mut rng = Rng::new(0x0F1A_5008);
    for _ in 0..40 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        for steps in [7usize, 40, 201] {
            let call = binomial_crr(s, k, t, r, sigma, q, steps, true, false).unwrap();
            let put = binomial_crr(s, k, t, r, sigma, q, steps, false, false).unwrap();
            let residue = put_call_parity_check(call, put, s, k, t, r, q).abs();
            assert!(residue < 1e-9 * s.max(k), "the binomial at {steps} steps left {residue}");
        }
    }
}

#[test]
fn prop_the_trinomial_lattice_is_arbitrage_free_only_in_the_limit() {
    // Matching the first two moments of the *log* price does not make the
    // price a martingale: it makes it one to O(dt^2). So the trinomial's
    // call and put violate parity by a residual that is real at coarse
    // step counts and falls as one over the square of the steps. Testing
    // the rate is a much stronger statement than testing a tolerance --
    // it says the violation is a discretisation artefact and not a bug.
    let mut rng = Rng::new(0x0F1A_5014);
    for _ in 0..20 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let residue = |steps: usize| {
            let call = trinomial(s, k, t, r, sigma, q, steps, true, false).unwrap();
            let put = trinomial(s, k, t, r, sigma, q, steps, false, false).unwrap();
            put_call_parity_check(call, put, s, k, t, r, q).abs()
        };
        let coarse = residue(25);
        let fine = residue(100);
        if coarse < 1e-12 * s {
            continue;
        }
        let ratio = coarse / fine;
        assert!(
            (10.0..24.0).contains(&ratio),
            "quadrupling the steps cut the parity residue by {ratio}, not sixteenfold"
        );
        // And by sixteen hundred steps it is gone for practical purposes.
        assert!(residue(1600) < 1e-6 * s.max(k));
    }
}

#[test]
fn prop_both_lattices_converge_to_the_closed_form() {
    let mut rng = Rng::new(0x0F1A_5009);
    for _ in 0..25 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let binomial = binomial_crr(s, k, t, r, sigma, q, 3000, true, false).unwrap();
        let tri = trinomial(s, k, t, r, sigma, q, 1500, true, false).unwrap();
        assert!(
            (binomial - exact).abs() < 1e-2 * s.max(1.0),
            "the binomial gave {binomial} against {exact}"
        );
        assert!(
            (tri - exact).abs() < 1e-2 * s.max(1.0),
            "the trinomial gave {tri} against {exact}"
        );
    }
}

#[test]
fn prop_early_exercise_is_worth_nothing_on_a_call_that_pays_no_dividend() {
    // And is worth something on every put. Both are theorems about the
    // exercise decision, independent of the numbers.
    let mut rng = Rng::new(0x0F1A_500A);
    for _ in 0..25 {
        let (s, k, t, r, sigma, _) = draw(&mut rng);
        if r <= 0.0 {
            continue;
        }
        let european = binomial_crr(s, k, t, r, sigma, 0.0, 800, true, false).unwrap();
        let american = binomial_crr(s, k, t, r, sigma, 0.0, 800, true, true).unwrap();
        assert!(
            (american - european).abs() < 1e-9 * s,
            "an American call gained {} by early exercise",
            american - european
        );
        let euro_put = binomial_crr(s, k, t, r, sigma, 0.0, 800, false, false).unwrap();
        let amer_put = binomial_crr(s, k, t, r, sigma, 0.0, 800, false, true).unwrap();
        assert!(amer_put >= euro_put - 1e-9 * s, "the American put was worth less");
        assert!(amer_put >= (k - s).max(0.0) - 1e-9 * s, "it was worth less than exercising");
    }
}

#[test]
fn prop_monte_carlo_lands_within_its_own_error_bar() {
    let mut rng = Rng::new(0x0F1A_500B);
    let mut outside = 0usize;
    let trials = 60;
    for _ in 0..trials {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let (price, error) =
            monte_carlo_european(s, k, t, r, sigma, q, true, 8_000, &mut rng).unwrap();
        assert!(error >= 0.0 && price >= 0.0);
        if (price - exact).abs() > 3.0 * error + 1e-9 * s {
            outside += 1;
        }
    }
    // Three standard errors should be exceeded a few times in a thousand,
    // not a few times in sixty.
    assert!(outside <= 2, "{outside} of {trials} draws missed by more than three errors");
}

#[test]
fn prop_a_knock_in_and_a_knock_out_partition_the_paths() {
    // On identical paths every one pays into exactly one of the two, so
    // the sum is the barrier-free price exactly rather than statistically.
    let mut rng = Rng::new(0x0F1A_500C);
    for _ in 0..15 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let seed = rng.next_u64();
        let price = |kind: Barrier, level: f64| {
            let mut inner = Rng::new(seed);
            monte_carlo_barrier(s, k, level, kind, t, r, sigma, q, true, 60, 8_000, &mut inner)
                .unwrap()
                .0
        };
        let vanilla = price(Barrier::UpAndOut, 1e12);
        for (out, into, level) in [
            (Barrier::UpAndOut, Barrier::UpAndIn, s * 1.2),
            (Barrier::DownAndOut, Barrier::DownAndIn, s * 0.8),
        ] {
            let dead = price(out, level);
            let alive = price(into, level);
            assert!(
                (dead + alive - vanilla).abs() < 1e-9 * s,
                "{dead} + {alive} against {vanilla}"
            );
            assert!(dead >= -1e-12 && alive >= -1e-12);
            assert!(dead <= vanilla + 1e-9 * s);
        }
    }
}

#[test]
fn prop_the_path_dependent_payoffs_sit_where_their_payoffs_put_them() {
    // A lookback call pays on the running maximum, which dominates the
    // terminal price path by path; an Asian pays on the average, which is
    // less variable than the terminal price. Both orderings are pathwise
    // and hold whatever the parameters.
    let mut rng = Rng::new(0x0F1A_500D);
    for _ in 0..12 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let european = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let (lookback, look_error) =
            monte_carlo_lookback(s, k, t, r, sigma, q, true, 60, 12_000, &mut rng).unwrap();
        assert!(
            lookback > european - 3.0 * look_error - 1e-9 * s,
            "the lookback at {lookback} was under the European {european}"
        );
        let (asian, asian_error) =
            monte_carlo_asian(s, k, t, r, sigma, q, true, 60, 12_000, &mut rng).unwrap();
        assert!(
            asian < european + 3.0 * asian_error + 1e-9 * s,
            "the Asian at {asian} was over the European {european}"
        );
        assert!(asian >= 0.0 && lookback >= 0.0);
    }
}

#[test]
fn prop_least_squares_monte_carlo_brackets_the_european_and_the_tree() {
    let mut rng = Rng::new(0x0F1A_500E);
    for _ in 0..8 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let european = black_scholes(s, k, t, r, sigma, q, false).unwrap();
        if european < 0.05 * s {
            continue;
        }
        let tree = binomial_crr(s, k, t, r, sigma, q, 1500, false, true).unwrap();
        let regressed =
            longstaff_schwartz_american(s, k, t, r, sigma, q, false, 40, 20_000, &mut rng).unwrap();
        assert!(
            regressed > european - 0.05 * european,
            "the regression gave {regressed} against a European {european}"
        );
        assert!(
            (regressed - tree).abs() < 0.05 * tree,
            "the regression gave {regressed} against the tree's {tree}"
        );
    }
}

#[test]
fn prop_the_general_models_reduce_to_the_special_one() {
    // Merton with no jumps and Heston with no volatility of volatility
    // must both be Black-Scholes. These are the strongest checks available
    // on the two, because the target is exact.
    let mut rng = Rng::new(0x0F1A_500F);
    for _ in 0..40 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        for call in [true, false] {
            let exact = black_scholes(s, k, t, r, sigma, q, call).unwrap();
            let jumpless = merton_jump_price(s, k, t, r, sigma, q, 0.0, 0.0, 0.0, call).unwrap();
            assert!(
                (jumpless - exact).abs() < 1e-11 * s,
                "Merton without jumps gave {jumpless} against {exact}"
            );
        }
    }
    for _ in 0..6 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        let exact = black_scholes(s, k, t, r, sigma, q, true).unwrap();
        let (heston, error) = heston_price_mc(
            s,
            k,
            t,
            r,
            q,
            sigma * sigma,
            2.0,
            sigma * sigma,
            1e-8,
            0.0,
            true,
            100,
            12_000,
            &mut rng,
        )
        .unwrap();
        assert!(
            (heston - exact).abs() < 3.0 * error + 1e-9 * s,
            "Heston without volatility of volatility gave {heston} +- {error} against {exact}"
        );
    }
}

#[test]
fn prop_jumps_only_ever_add_value_to_an_option() {
    // A jump component adds variance to the terminal distribution at the
    // same forward, and an option is a convex payoff, so its price cannot
    // fall. The compensator is what keeps the forward fixed, and this is
    // the property that catches it being wrong.
    let mut rng = Rng::new(0x0F1A_5010);
    for _ in 0..60 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        for call in [true, false] {
            let plain = black_scholes(s, k, t, r, sigma, q, call).unwrap();
            let jumped =
                merton_jump_price(s, k, t, r, sigma, q, 0.4, -0.08, 0.2, call).unwrap();
            assert!(
                jumped > plain - 1e-9 * s,
                "jumps took {} off an option worth {plain}",
                plain - jumped
            );
        }
        // And the jump prices still satisfy parity, since the underlying
        // distribution is the same for both sides.
        let call = merton_jump_price(s, k, t, r, sigma, q, 0.4, -0.08, 0.2, true).unwrap();
        let put = merton_jump_price(s, k, t, r, sigma, q, 0.4, -0.08, 0.2, false).unwrap();
        assert!(
            put_call_parity_check(call, put, s, k, t, r, q).abs() < 1e-9 * s.max(k),
            "the jump model broke parity"
        );
    }
}

#[test]
fn prop_the_grid_prices_what_the_formula_does() {
    let mut rng = Rng::new(0x0F1A_5011);
    for _ in 0..20 {
        let (s, k, t, r, sigma, q) = draw(&mut rng);
        for call in [true, false] {
            let exact = black_scholes(s, k, t, r, sigma, q, call).unwrap();
            let grid =
                bs_pde_crank_nicolson(s, k, t, r, sigma, q, call, false, 401, 200).unwrap();
            assert!(
                (grid - exact).abs() < 5e-3 * s.max(1.0),
                "S={s} K={k} T={t} vol={sigma}: the grid gave {grid} against {exact}"
            );
        }
        // The American value is never below the European one, and never
        // below exercising now.
        let american = bs_pde_crank_nicolson(s, k, t, r, sigma, q, false, true, 401, 200).unwrap();
        let european = black_scholes(s, k, t, r, sigma, q, false).unwrap();
        assert!(american > european - 5e-3 * s.max(1.0));
        assert!(american >= (k - s).max(0.0) - 1e-9 * s);
    }
}

#[test]
fn prop_svi_produces_a_positive_variance_with_linear_wings() {
    let mut rng = Rng::new(0x0F1A_5012);
    for _ in 0..200 {
        let params = Svi {
            a: 0.001 + 0.1 * rng.next_f64(),
            b: 0.01 + 0.4 * rng.next_f64(),
            rho: -0.95 + 1.9 * rng.next_f64(),
            m: -0.3 + 0.6 * rng.next_f64(),
            sigma: 0.02 + 0.4 * rng.next_f64(),
        };
        // The minimum sits at a + b sigma sqrt(1 - rho^2), so a positive
        // `a` is enough to keep the whole curve above zero.
        for step in 0..21 {
            let k = -1.0 + 0.1 * step as f64;
            let variance = volatility_smile_svi(&params, k).unwrap();
            assert!(variance > 0.0 && variance.is_finite(), "at k={k} variance was {variance}");
        }
        // Far out, the slopes are exactly b(1 +- rho).
        let far = 1e4;
        let right = volatility_smile_svi(&params, far + 1.0).unwrap()
            - volatility_smile_svi(&params, far).unwrap();
        assert!((right - params.b * (1.0 + params.rho)).abs() < 1e-6, "right wing slope {right}");
        let left = volatility_smile_svi(&params, -far - 1.0).unwrap()
            - volatility_smile_svi(&params, -far).unwrap();
        assert!((left - params.b * (1.0 - params.rho)).abs() < 1e-6, "left wing slope {left}");
    }
}

#[test]
fn prop_the_svi_fit_reproduces_the_curve_it_was_shown() {
    // The parameters need not come back -- b and sigma trade off against
    // each other -- but the *shape* must, which is what a smile is for.
    let mut rng = Rng::new(0x0F1A_5013);
    for _ in 0..20 {
        let truth = Svi {
            a: 0.005 + 0.05 * rng.next_f64(),
            b: 0.05 + 0.2 * rng.next_f64(),
            rho: -0.8 + 1.0 * rng.next_f64(),
            m: -0.1 + 0.2 * rng.next_f64(),
            sigma: 0.05 + 0.2 * rng.next_f64(),
        };
        let strikes: Vec<f64> = (0..15).map(|i| -0.7 + 0.1 * i as f64).collect();
        let variances: Vec<f64> =
            strikes.iter().map(|k| volatility_smile_svi(&truth, *k).unwrap()).collect();
        let fitted = svi_fit(&strikes, &variances).unwrap();
        let scale = variances.iter().fold(0.0f64, |a, b| a.max(*b));
        for k in &strikes {
            let want = volatility_smile_svi(&truth, *k).unwrap();
            let got = volatility_smile_svi(&fitted, *k).unwrap();
            assert!((got - want).abs() < 1e-3 * scale, "at k={k}: {got} against {want}");
        }
    }
}

