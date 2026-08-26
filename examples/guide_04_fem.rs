//! Guide, chapter 4: solving a differential equation, and proving the
//! answer converges at the rate the theory predicts.
//!
//! Run with `cargo run --example guide_04_fem`. CI runs it too.

use rust_physics_engine::fem::fem1d::{
    convergence_rate, fem_1d_error_h1_seminorm, fem_1d_error_l2, fem_1d_poisson,
    fem_1d_quadratic, Bc, Fem1dSolution,
};
use std::f64::consts::PI;

fn main() {
    // Solve  −u″ = f  on [0, 1] with u(0) = u(1) = 0.
    //
    // Choosing f = π² sin(πx) means the exact answer is u = sin(πx), which
    // is what makes the error measurable rather than merely plausible. A
    // solver you cannot check against a closed form is a solver you are
    // trusting rather than testing.
    let f = |x: f64| PI * PI * (PI * x).sin();
    let exact = |x: f64| (PI * x).sin();
    let d_exact = |x: f64| PI * (PI * x).cos();

    println!("−u\u{2033} = \u{3c0}\u{b2}sin(\u{3c0}x) on [0,1], u(0) = u(1) = 0");
    println!("exact solution u = sin(\u{3c0}x)\n");

    // Refine the mesh and watch the error fall.
    let counts = [8usize, 16, 32, 64, 128];
    let mut hs = Vec::new();
    let mut l2 = Vec::new();
    let mut h1 = Vec::new();

    println!("  P1 elements");
    println!("  {:>6}  {:>6}  {:>12}  {:>12}", "cells", "h", "L2 error", "H1 error");
    for &n in &counts {
        let values = fem_1d_poisson(&f, 0.0, 1.0, (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)), n)
            .expect("the Poisson problem is well posed");
        let solution = Fem1dSolution::new(0.0, 1.0, 1, values).expect("nodal values fit P1");

        let h = 1.0 / n as f64;
        let e_l2 = fem_1d_error_l2(&solution, &exact);
        let e_h1 = fem_1d_error_h1_seminorm(&solution, &d_exact);
        println!("  {n:>6}  {h:>6.4}  {e_l2:>12.3e}  {e_h1:>12.3e}");
        hs.push(h);
        l2.push(e_l2);
        h1.push(e_h1);
    }

    // The rate is the slope of log(error) against log(h). For linear
    // elements the theory says 2 in L2 and 1 in H1 -- one order is lost to
    // differentiating, because the energy norm measures the derivative.
    let rate_l2 = convergence_rate(&l2, &hs).expect("enough refinements");
    let rate_h1 = convergence_rate(&h1, &hs).expect("enough refinements");
    println!("\n  measured rate   L2 {rate_l2:.2}   H1 {rate_h1:.2}");
    println!("  theory          L2 2.00   H1 1.00");
    assert!((rate_l2 - 2.0).abs() < 0.1, "L2 rate {rate_l2} is not 2");
    assert!((rate_h1 - 1.0).abs() < 0.1, "H1 rate {rate_h1} is not 1");

    // Quadratic elements buy an order in each norm for the same mesh.
    let one = |_: f64| 1.0;
    let zero = |_: f64| 0.0;
    let mut hs2 = Vec::new();
    let mut l2_p2 = Vec::new();

    println!("\n  P2 elements");
    println!("  {:>6}  {:>6}  {:>12}", "cells", "h", "L2 error");
    for &n in &counts[..4] {
        let values = fem_1d_quadratic(
            &one,
            &zero,
            &f,
            0.0,
            1.0,
            (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)),
            n,
        )
        .expect("well posed");
        let solution = Fem1dSolution::new(0.0, 1.0, 2, values).expect("nodal values fit P2");
        let h = 1.0 / n as f64;
        let e = fem_1d_error_l2(&solution, &exact);
        println!("  {n:>6}  {h:>6.4}  {e:>12.3e}");
        hs2.push(h);
        l2_p2.push(e);
    }
    let rate_p2 = convergence_rate(&l2_p2, &hs2).expect("enough refinements");
    println!("\n  measured rate   L2 {rate_p2:.2}");
    println!("  theory          L2 3.00");
    assert!((rate_p2 - 3.0).abs() < 0.15, "P2 L2 rate {rate_p2} is not 3");

    // What makes this a *proof* rather than a plot is that the rate is
    // predicted before it is measured. An error that merely shrinks tells
    // you nothing; an error that shrinks at exactly h² tells you the
    // discretisation is the one you think it is.
    println!("\nboth rates match the theory, so the discretisation is correct");
}
