//! Properties of the interest-rate module.
//!
//! Rates arithmetic is almost entirely identities, which makes it unusually
//! testable. A conversion between compounding conventions must preserve
//! the growth factor. A yield must reproduce the price it was solved from.
//! A bootstrapped curve must reprice the bonds it was stripped from. A
//! forward rate must make rolling equal to holding. Duration and convexity
//! must be the derivatives they are named after.
//!
//! Where a quantity is *defined* as the solution to an equation --
//! internal rate of return, yield to maturity, the stripped zero rate --
//! the sharp test is to substitute the answer back, and that is what most
//! of these do.

use rust_physics_engine::finance::rates::{
    amortization_schedule, bond_price, bootstrap_zero_curve, cir_bond_price,
    cir_feller_condition, convexity, discount_factor, duration_macaulay, duration_modified,
    equivalent_rate, forward_rate, irr, mortgage_payment, nelson_siegel, npv, ns_fit,
    vasicek_bond_price, xirr, Compounding, CurveBond,
};
use rust_physics_engine::monte_carlo::Rng;

const CONVENTIONS: [Compounding; 5] = [
    Compounding::Annual,
    Compounding::SemiAnnual,
    Compounding::Quarterly,
    Compounding::Monthly,
    Compounding::Continuous,
];

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

#[test]
fn prop_converting_a_rate_preserves_every_discount_factor_it_implies() {
    // The quoted number changes and the money does not. A round trip must
    // return the rate, and the converted rate must discount identically at
    // every horizon, not only at one year.
    let mut rng = Rng::new(0x0F1B_1001);
    for _ in 0..400 {
        let rate = -0.1 + 2.0 * rng.next_f64();
        let from = CONVENTIONS[pick(&mut rng, 5)];
        let to = CONVENTIONS[pick(&mut rng, 5)];
        let moved = equivalent_rate(rate, from, to).unwrap();
        let back = equivalent_rate(moved, to, from).unwrap();
        assert!((back - rate).abs() < 1e-11, "{from:?}->{to:?} at {rate} came back {back}");
        for t in [0.03f64, 1.0, 4.5, 40.0] {
            let here = discount_factor(rate, t, from).unwrap();
            let there = discount_factor(moved, t, to).unwrap();
            assert!(
                (here - there).abs() < 1e-12 * here.max(1.0),
                "at t={t}: {here} against {there}"
            );
        }
    }
}

#[test]
fn prop_a_discount_factor_behaves_like_one() {
    // Between zero and one for a positive rate, one at zero time,
    // decreasing in both the rate and the horizon, and multiplicative
    // across horizons under continuous compounding.
    let mut rng = Rng::new(0x0F1B_1002);
    for _ in 0..300 {
        let rate = 0.0001 + 0.5 * rng.next_f64();
        let convention = CONVENTIONS[pick(&mut rng, 5)];
        assert!((discount_factor(rate, 0.0, convention).unwrap() - 1.0).abs() < 1e-15);
        let mut previous = 1.0 + 1e-15;
        for t in [0.1f64, 1.0, 5.0, 30.0] {
            let factor = discount_factor(rate, t, convention).unwrap();
            assert!((0.0..=1.0).contains(&factor), "at t={t} the factor was {factor}");
            assert!(factor < previous, "the factor rose at t={t}");
            previous = factor;
            // A higher rate discounts harder.
            assert!(discount_factor(rate * 1.5, t, convention).unwrap() < factor);
        }
        // Continuous compounding makes rates additive across horizons.
        let a = discount_factor(rate, 2.0, Compounding::Continuous).unwrap();
        let b = discount_factor(rate, 3.0, Compounding::Continuous).unwrap();
        let both = discount_factor(rate, 5.0, Compounding::Continuous).unwrap();
        assert!((a * b - both).abs() < 1e-15);
    }
}

#[test]
fn prop_an_internal_rate_of_return_zeroes_the_value_it_was_found_from() {
    // Substituting the answer back is the definition, and the only check
    // that does not assume the solver's own machinery.
    let mut rng = Rng::new(0x0F1B_1003);
    let mut solved = 0usize;
    for _ in 0..300 {
        let n = 2 + pick(&mut rng, 12);
        let outlay = -(100.0 + 900.0 * rng.next_f64());
        let mut flows = vec![outlay];
        for _ in 1..n {
            flows.push(10.0 + 300.0 * rng.next_f64());
        }
        let Some(rate) = irr(&flows).unwrap() else { continue };
        solved += 1;
        assert!(rate > -1.0 && rate.is_finite());
        let value = npv(rate, &flows).unwrap();
        assert!(value.abs() < 1e-6 * outlay.abs(), "the value at its own rate was {value}");
        // Value falls in the rate here, since every flow after the first
        // is positive.
        assert!(npv(rate - 0.01, &flows).unwrap() > 0.0);
        assert!(npv(rate + 0.01, &flows).unwrap() < 0.0);
    }
    assert!(solved > 250, "only {solved} of 300 draws had a single sign change");
}

#[test]
fn prop_xirr_agrees_with_irr_on_whole_periods_and_zeroes_its_own_value() {
    let mut rng = Rng::new(0x0F1B_1004);
    for _ in 0..150 {
        let n = 3 + pick(&mut rng, 8);
        let mut flows = vec![-(100.0 + 900.0 * rng.next_f64())];
        for _ in 1..n {
            flows.push(10.0 + 300.0 * rng.next_f64());
        }
        let whole: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let Some(annual) = irr(&flows).unwrap() else { continue };
        let matched = xirr(&whole, &flows).unwrap().expect("the same single sign change");
        assert!((matched - annual).abs() < 1e-8, "{matched} against {annual}");

        // On irregular dates the answer still zeroes its own value.
        let mut times = vec![0.0f64];
        for _ in 1..n {
            times.push(times.last().unwrap() + 0.1 + 1.5 * rng.next_f64());
        }
        let Some(rate) = xirr(&times, &flows).unwrap() else { continue };
        let value: f64 =
            times.iter().zip(flows.iter()).map(|(t, c)| c * (1.0 + rate).powf(-t)).sum();
        assert!(value.abs() < 1e-6 * flows[0].abs(), "the value was {value}");
    }
}

#[test]
fn prop_a_bond_prices_at_par_exactly_when_its_coupon_is_its_yield() {
    let mut rng = Rng::new(0x0F1B_1005);
    for _ in 0..400 {
        let rate = 0.0005 + 0.4 * rng.next_f64();
        let periods = 1 + pick(&mut rng, 120);
        let face = 10.0 + 990.0 * rng.next_f64();
        let par = bond_price(face, face * rate, rate, periods).unwrap();
        assert!((par - face).abs() < 1e-9 * face, "it priced at {par} against a face of {face}");
        // Above the yield it trades over par and below it under, with no
        // exceptions.
        let over = bond_price(face, face * rate * 1.3, rate, periods).unwrap();
        let under = bond_price(face, face * rate * 0.7, rate, periods).unwrap();
        assert!(over > face && under < face, "{over} and {under} against {face}");
    }
}

#[test]
fn prop_the_price_falls_monotonically_in_the_yield() {
    // Which is what makes the yield unique, and is the difference between
    // a bond and a project with alternating cashflows.
    let mut rng = Rng::new(0x0F1B_1006);
    for _ in 0..200 {
        let periods = 1 + pick(&mut rng, 60);
        let coupon = 20.0 * rng.next_f64();
        let mut previous = f64::INFINITY;
        for step in 0..14 {
            let ytm = -0.08 + 0.05 * step as f64;
            let price = bond_price(100.0, coupon, ytm, periods).unwrap();
            assert!(price < previous, "the price rose at a yield of {ytm}");
            assert!(price > 0.0);
            previous = price;
        }
    }
}

#[test]
fn prop_duration_and_convexity_are_the_derivatives_of_the_price() {
    // Modified duration is -(1/P) dP/dy and convexity is (1/P) d2P/dy2.
    // The second needs Richardson extrapolation: dividing by h^2 amplifies
    // the cancellation, so a small step is dominated by round-off and a
    // large one by truncation.
    let mut rng = Rng::new(0x0F1B_1007);
    for _ in 0..250 {
        let periods = 1 + pick(&mut rng, 80);
        let coupon = 15.0 * rng.next_f64();
        let ytm = -0.03 + 0.25 * rng.next_f64();
        let price = |y: f64| bond_price(100.0, coupon, y, periods).unwrap();
        let base = price(ytm);
        if base < 1e-3 {
            continue;
        }
        let h = 1e-6;
        let modified = duration_modified(100.0, coupon, ytm, periods).unwrap();
        let first = -(price(ytm + h) - price(ytm - h)) / (2.0 * h) / base;
        assert!(
            (modified - first).abs() < 1e-5 * modified.abs().max(1.0),
            "modified duration {modified} against {first}"
        );
        let macaulay = duration_macaulay(100.0, coupon, ytm, periods).unwrap();
        assert!((macaulay - modified * (1.0 + ytm)).abs() < 1e-12 * macaulay.max(1.0));
        assert!(macaulay > 0.0 && macaulay <= periods as f64 + 1e-9);

        let hc = 1e-3;
        let second = |h: f64| (price(ytm + h) - 2.0 * base + price(ytm - h)) / (h * h) / base;
        let extrapolated = (4.0 * second(0.5 * hc) - second(hc)) / 3.0;
        let convex = convexity(100.0, coupon, ytm, periods).unwrap();
        assert!(convex > 0.0, "convexity was {convex}");
        assert!(
            (convex - extrapolated).abs() < 1e-3 * convex,
            "convexity {convex} against {extrapolated}"
        );
    }
}

#[test]
fn prop_a_coupon_can_only_shorten_duration() {
    // Duration is a discounted-cashflow-weighted average time, so adding
    // weight at earlier dates moves the centre of mass earlier. A
    // zero-coupon bond is the extreme, at exactly its maturity.
    let mut rng = Rng::new(0x0F1B_1008);
    for _ in 0..200 {
        let periods = 1 + pick(&mut rng, 60);
        let ytm = 0.001 + 0.2 * rng.next_f64();
        let zero = duration_macaulay(100.0, 0.0, ytm, periods).unwrap();
        assert!((zero - periods as f64).abs() < 1e-9, "a zero-coupon duration was {zero}");
        let mut previous = zero + 1e-12;
        for coupon in [0.5f64, 2.0, 6.0, 15.0] {
            let duration = duration_macaulay(100.0, coupon, ytm, periods).unwrap();
            assert!(duration < previous, "a coupon of {coupon} lengthened duration");
            previous = duration;
        }
    }
}

#[test]
fn prop_bootstrapping_reprices_the_bonds_it_was_given() {
    // The curve is defined as the one that reproduces the quotes, so
    // repricing them is the definition rather than a tolerance. A flat
    // curve is the case with an answer known in advance.
    let mut rng = Rng::new(0x0F1B_1009);
    for _ in 0..60 {
        let level = 0.001 + 0.08 * rng.next_f64();
        let slope = -0.03 + 0.06 * rng.next_f64();
        let truth = |t: f64| level + slope * (1.0 - (-t / 2.5).exp());
        let coupon = 0.12 * rng.next_f64();
        let bonds: Vec<CurveBond> = (1..=7)
            .map(|years| {
                let mut price = 0.0;
                for period in 1..=years {
                    let t = period as f64;
                    price += coupon * (-truth(t) * t).exp();
                }
                price += (-truth(years as f64) * years as f64).exp();
                CurveBond { maturity: years as f64, coupon, price, frequency: 1.0 }
            })
            .collect();
        let Ok(curve) = bootstrap_zero_curve(&bonds) else { continue };
        assert_eq!(curve.len(), bonds.len());
        for (t, zero) in &curve {
            assert!(
                (zero - truth(*t)).abs() < 1e-10,
                "at t={t} the strip gave {zero} against {}",
                truth(*t)
            );
        }
        // Repricing each bond off the recovered curve returns its quote.
        for bond in &bonds {
            let years = bond.maturity as usize;
            let mut price = 0.0;
            for period in 1..=years {
                let t = period as f64;
                let zero = curve.iter().find(|(m, _)| (m - t).abs() < 1e-9).expect("a node").1;
                price += bond.coupon * (-zero * t).exp();
            }
            let zero = curve[years - 1].1;
            price += (-zero * bond.maturity).exp();
            assert!(
                (price - bond.price).abs() < 1e-10,
                "the {years}-year bond repriced at {price} against {}",
                bond.price
            );
        }
    }
}

#[test]
fn prop_a_forward_rate_makes_rolling_the_same_as_holding() {
    // Investing to the far date must equal investing to the near one and
    // rolling at the forward. It is a no-arbitrage identity, so it holds
    // for any two rates and dates whatsoever.
    let mut rng = Rng::new(0x0F1B_100A);
    for _ in 0..500 {
        let t1 = 5.0 * rng.next_f64();
        let t2 = t1 + 0.01 + 20.0 * rng.next_f64();
        let z1 = -0.02 + 0.15 * rng.next_f64();
        let z2 = -0.02 + 0.15 * rng.next_f64();
        let forward = forward_rate(z1, t1, z2, t2).unwrap();
        let rolled = (-z1 * t1).exp() * (-forward * (t2 - t1)).exp();
        let direct = (-z2 * t2).exp();
        assert!(
            (rolled - direct).abs() < 1e-12 * direct.max(1.0),
            "{rolled} against {direct}"
        );
        // A flat curve implies a forward equal to the level.
        assert!((forward_rate(z1, t1, z1, t2).unwrap() - z1).abs() < 1e-12);
        // A rising curve implies a forward above the far rate.
        if z2 > z1 && t1 > 0.0 {
            assert!(forward > z2, "a rising curve gave a forward of {forward} under {z2}");
        }
    }
}

#[test]
fn prop_nelson_siegel_is_bounded_by_its_own_level_and_slope() {
    // The curve runs from b0 + b1 at the short end to b0 at the long one,
    // and the curvature term is what it does in between. Both limits are
    // exact and neither depends on tau.
    let mut rng = Rng::new(0x0F1B_100B);
    for _ in 0..300 {
        let b0 = -0.02 + 0.12 * rng.next_f64();
        let b1 = -0.06 + 0.12 * rng.next_f64();
        let b2 = -0.06 + 0.12 * rng.next_f64();
        let tau = 0.1 + 8.0 * rng.next_f64();
        assert!((nelson_siegel(0.0, b0, b1, b2, tau).unwrap() - (b0 + b1)).abs() < 1e-15);
        assert!((nelson_siegel(1e-10, b0, b1, b2, tau).unwrap() - (b0 + b1)).abs() < 1e-9);
        assert!((nelson_siegel(1e10, b0, b1, b2, tau).unwrap() - b0).abs() < 1e-8);
        for t in [0.01f64, 0.5, 3.0, 12.0, 50.0] {
            let rate = nelson_siegel(t, b0, b1, b2, tau).unwrap();
            assert!(rate.is_finite());
            // The whole curve lies within the span the three coefficients
            // can reach.
            let reach = b0.abs() + b1.abs() + b2.abs();
            assert!(rate.abs() <= reach + 1e-12, "at t={t} the rate was {rate}");
        }
        // Rescaling tau and t together leaves the curve alone, since the
        // model depends on them only through t/tau.
        let factor = 0.5 + 3.0 * rng.next_f64();
        for t in [0.3f64, 2.0, 9.0] {
            let here = nelson_siegel(t, b0, b1, b2, tau).unwrap();
            let there = nelson_siegel(factor * t, b0, b1, b2, factor * tau).unwrap();
            assert!((here - there).abs() < 1e-13, "{here} against {there}");
        }
    }
}

#[test]
fn prop_the_nelson_siegel_fit_reproduces_a_curve_from_its_own_family() {
    let mut rng = Rng::new(0x0F1B_100C);
    let maturities = [0.25f64, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
    for _ in 0..40 {
        let b0 = 0.01 + 0.06 * rng.next_f64();
        let b1 = -0.05 + 0.06 * rng.next_f64();
        let b2 = -0.04 + 0.08 * rng.next_f64();
        let tau = 0.5 + 5.0 * rng.next_f64();
        let yields: Vec<f64> =
            maturities.iter().map(|t| nelson_siegel(*t, b0, b1, b2, tau).unwrap()).collect();
        let (f0, f1, f2, ftau) = ns_fit(&maturities, &yields).unwrap();
        for t in [0.1f64, 0.75, 4.0, 15.0, 25.0, 40.0] {
            let want = nelson_siegel(t, b0, b1, b2, tau).unwrap();
            let got = nelson_siegel(t, f0, f1, f2, ftau).unwrap();
            assert!((got - want).abs() < 5e-5, "at t={t}: {got} against {want}");
        }
    }
}

#[test]
fn prop_a_short_rate_bond_price_behaves_like_a_bond_price() {
    // One at maturity, positive everywhere, falling with maturity, and
    // rising with volatility because discounting is convex.
    //
    // Staying *below* one is a CIR property and not a Vasicek one. A
    // Gaussian short rate can go negative, and with weak mean reversion
    // the convexity term `sigma^2/(2 kappa^2)` can exceed `theta`
    // outright, so Vasicek's long yield is negative and its bond price
    // exceeds one. That is the model's known feature, not an error, so it
    // is asserted where it applies and demonstrated where it does not.
    let mut rng = Rng::new(0x0F1B_100D);
    for _ in 0..250 {
        let r0 = 0.001 + 0.1 * rng.next_f64();
        let kappa = 0.05 + 2.0 * rng.next_f64();
        let theta = 0.001 + 0.1 * rng.next_f64();
        let sigma = 0.001 + 0.05 * rng.next_f64();
        assert!((vasicek_bond_price(r0, kappa, theta, sigma, 0.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((cir_bond_price(r0, kappa, theta, sigma, 0.0).unwrap() - 1.0).abs() < 1e-15);
        let mut cir_previous = 1.0 + 1e-15;
        for t in [0.05f64, 1.0, 7.0, 25.0] {
            let vasicek = vasicek_bond_price(r0, kappa, theta, sigma, t).unwrap();
            let cir = cir_bond_price(r0, kappa, theta, sigma, t).unwrap();
            assert!(vasicek > 0.0 && vasicek.is_finite(), "Vasicek gave {vasicek} at t={t}");
            // CIR's square-root diffusion keeps the rate non-negative, so
            // its bond can never be worth more than the unit it pays.
            assert!(cir > 0.0 && cir < 1.0, "CIR gave {cir} at t={t}");
            assert!(cir < cir_previous, "the CIR price rose at t={t}");
            cir_previous = cir;
            // More volatility, more value, in both models.
            assert!(vasicek_bond_price(r0, kappa, theta, sigma * 2.0, t).unwrap() > vasicek);
            assert!(cir_bond_price(r0, kappa, theta, sigma * 2.0, t).unwrap() > cir);
        }
        // The Feller condition is a statement about the parameters alone.
        assert_eq!(
            cir_feller_condition(kappa, theta, sigma),
            2.0 * kappa * theta >= sigma * sigma
        );
    }
}

#[test]
fn prop_a_gaussian_short_rate_can_make_a_bond_worth_more_than_it_pays() {
    // Weak mean reversion and appreciable volatility put Vasicek's
    // long-run yield `theta - sigma^2/(2 kappa^2)` below zero, and a bond
    // discounted at a negative yield is worth more than its face. CIR
    // cannot do this at any parameters, which is the whole point of the
    // square-root diffusion.
    let (r0, kappa, theta, sigma) = (0.005, 0.05, 0.01, 0.03);
    let long_yield = theta - sigma * sigma / (2.0 * kappa * kappa);
    assert!(long_yield < 0.0, "the convexity term did not overwhelm theta: {long_yield}");
    let vasicek = vasicek_bond_price(r0, kappa, theta, sigma, 30.0).unwrap();
    assert!(vasicek > 1.0, "the Vasicek bond was worth only {vasicek}");
    let cir = cir_bond_price(r0, kappa, theta, sigma, 30.0).unwrap();
    assert!(cir < 1.0, "the CIR bond was worth {cir}");
}

#[test]
fn prop_a_deterministic_rate_gives_both_models_the_same_price() {
    // With no diffusion the rate follows an ODE whose integral has a
    // closed form, and both affine models must land on it exactly. This
    // is what catches a mis-set A(t) factor, which the price alone would
    // look perfectly plausible without.
    let mut rng = Rng::new(0x0F1B_100E);
    for _ in 0..200 {
        let r0 = 0.001 + 0.1 * rng.next_f64();
        let kappa = 0.05 + 2.0 * rng.next_f64();
        let theta = 0.001 + 0.1 * rng.next_f64();
        for t in [0.1f64, 2.0, 15.0, 40.0] {
            let integral = theta * t + (r0 - theta) * (1.0 - (-kappa * t).exp()) / kappa;
            let expected = (-integral).exp();
            let vasicek = vasicek_bond_price(r0, kappa, theta, 0.0, t).unwrap();
            let cir = cir_bond_price(r0, kappa, theta, 0.0, t).unwrap();
            assert!((vasicek - expected).abs() < 1e-13, "Vasicek {vasicek} against {expected}");
            assert!((cir - expected).abs() < 1e-13, "CIR {cir} against {expected}");
        }
    }
}

#[test]
fn prop_a_level_payment_clears_the_loan_and_nothing_more() {
    let mut rng = Rng::new(0x0F1B_100F);
    for _ in 0..200 {
        let principal = 100.0 + 900_000.0 * rng.next_f64();
        let rate = 0.3 * rng.next_f64() / 12.0;
        let n = 1 + pick(&mut rng, 480);
        let payment = mortgage_payment(principal, rate, n).unwrap();
        assert!(payment > 0.0 && payment.is_finite());
        // The payment covers at least the principal per period, and at
        // least the first period's interest.
        assert!(payment >= principal / n as f64 - 1e-9);
        assert!(payment > principal * rate - 1e-9 || n == 1);

        let schedule = amortization_schedule(principal, rate, n).unwrap();
        assert_eq!(schedule.len(), n);
        assert!(schedule.last().unwrap().3.abs() < 1e-9, "a balance was left");
        let repaid: f64 = schedule.iter().map(|row| row.2).sum();
        assert!(
            (repaid - principal).abs() < 1e-6 * principal.max(1.0),
            "it repaid {repaid} of {principal}"
        );
        let mut balance = principal;
        for row in &schedule {
            assert!((row.1 - balance * rate).abs() < 1e-6 * principal.max(1.0) || row.3 == 0.0);
            assert!((row.1 + row.2 - row.0).abs() < 1e-9 * payment.max(1.0));
            assert!(row.3 <= balance + 1e-9, "the balance rose");
            assert!(row.3 >= -1e-9, "the balance went negative");
            balance = row.3;
        }
        // Total interest is total payments less the principal, and it is
        // never negative for a non-negative rate.
        let paid: f64 = schedule.iter().map(|row| row.0).sum();
        assert!(paid >= principal - 1e-6 * principal.max(1.0), "it paid back only {paid}");
    }
}
