//! Properties for `monte_carlo::quasi`.

use rust_physics_engine::monte_carlo::{mc_integrate_sobol, Rng, Sobol};

/// Sobol integration error on ∫_[0,1]^4 Π xᵢ = 1/16 shrinks faster
/// than pseudo-random sampling: the fitted log-log slope over
/// n = 1e3, 1e4, 1e5 is steeper than −0.5.
#[test]
fn prop_sobol_converges_faster_than_pseudo_random() {
    let f = |x: &[f64]| x.iter().product::<f64>();
    let exact = 1.0 / 16.0;
    let ns = [1_000usize, 10_000, 100_000];

    let sobol_errs: Vec<f64> = ns
        .iter()
        .map(|&n| (mc_integrate_sobol(&f, 4, n) - exact).abs().max(1e-16))
        .collect();

    // Log-log slope by least squares over the three points.
    let slope = |errs: &[f64]| {
        let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
        let ys: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
        let xm = xs.iter().sum::<f64>() / 3.0;
        let ym = ys.iter().sum::<f64>() / 3.0;
        let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - xm) * (y - ym)).sum();
        let den: f64 = xs.iter().map(|x| (x - xm) * (x - xm)).sum();
        num / den
    };
    let sobol_slope = slope(&sobol_errs);
    assert!(sobol_slope < -0.5, "Sobol slope {sobol_slope} not steeper than -0.5");

    // And at every n, Sobol beats an averaged pseudo-random error.
    let mut rng = Rng::new(111);
    for (&n, sobol_err) in ns.iter().zip(&sobol_errs) {
        let mut mc_err_sum = 0.0;
        let reps = 5;
        for _ in 0..reps {
            let mut sum = 0.0;
            for _ in 0..n {
                let p: Vec<f64> = (0..4).map(|_| rng.next_f64()).collect();
                sum += f(&p);
            }
            mc_err_sum += (sum / n as f64 - exact).abs();
        }
        let mc_err = mc_err_sum / reps as f64;
        assert!(
            *sobol_err < mc_err,
            "n={n}: sobol {sobol_err} not below pseudo-random {mc_err}"
        );
    }
}

/// Sobol nets are balanced: the prefix x₀..x₆₃ puts exactly half the
/// points in each half-interval of every axis. The sequence skips the
/// origin x₀ (which lies in the low half of every axis), so points
/// x₁..x₆₃ must contain exactly 31 low-half points per axis.
#[test]
fn prop_sobol_net_balance() {
    for dim in [2usize, 5, 13, 21] {
        let mut s = Sobol::new(dim);
        let pts: Vec<Vec<f64>> = (0..63).map(|_| s.next()).collect();
        for d in 0..dim {
            let low = pts.iter().filter(|p| p[d] < 0.5).count();
            assert_eq!(low, 31, "dim {dim}, axis {d} unbalanced");
        }
    }
}
