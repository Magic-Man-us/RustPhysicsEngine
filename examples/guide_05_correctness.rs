//! Guide, chapter 5: the tools for not being wrong.
//!
//! Dimensions checked in the type, arithmetic without rounding, and
//! Buckingham's theorem as an exact null space.
//!
//! Run with `cargo run --example guide_05_correctness`. CI runs it too.

use rust_physics_engine::exact::rational::Rational;
use rust_physics_engine::exact::symbolic::Expr;
use rust_physics_engine::units::dimensional::{buckingham_pi, dimensional_check_formula};
use rust_physics_engine::units::quantity::{parse_quantity, unit_convert, Dim, Quantity};

fn main() {
    // ---- 1. dimensions travel with the value -------------------------
    //
    // The Mars Climate Orbiter was lost to arithmetic a computer performed
    // correctly on numbers that meant something other than the receiving
    // code assumed. A Quantity carries seven exponents, so the mistake
    // becomes a type error instead of a trajectory.
    let force = Quantity::newtons(4.45);
    let distance = Quantity::meters(2.0);
    let work = force.mul(&distance).expect("force times distance");

    println!("dimensions");
    println!("  4.45 N x 2 m = {:.2} {}", work.value, work.dim);
    assert_eq!(work.dim, Dim::new(2, 1, -2, 0, 0, 0, 0)); // joules, exactly

    let time = Quantity::seconds(3.0);
    println!("  adding a force to a time -> {}", force.add(&time).unwrap_err());
    assert!(force.add(&time).is_err());

    // Square roots only exist when every exponent is even, which is a
    // refusal rather than a rounding decision: there is no square root of
    // a metre.
    let area = Quantity::new(9.0, Dim::new(2, 0, 0, 0, 0, 0, 0));
    println!("  sqrt(9 m^2)  = {:.1} {}", area.sqrt().unwrap().value, area.sqrt().unwrap().dim);
    assert!(Quantity::meters(9.0).sqrt().is_err());

    // Parsing and conversion, for when the units arrive as text.
    let g = parse_quantity("9.81 m/s^2").expect("a readable quantity");
    println!("  \"9.81 m/s^2\" parses to {} {}", g.value, g.dim);
    let mps = unit_convert(3.6, "km/h", "m/s").expect("same dimension");
    println!("  3.6 km/h     = {mps:.1} m/s");
    assert!((mps - 1.0).abs() < 1e-12);

    // ---- 2. checking a formula, not a number -------------------------
    //
    // Both sides of `x + v` are perfectly good floats, so no amount of
    // running the formula finds the mistake. Walking the expression does.
    let vars = [
        ("l", Dim::LENGTH),
        ("g", Dim::new(1, 0, -2, 0, 0, 0, 0)),
        ("t", Dim::TIME),
        ("omega", Dim::new(0, 0, -1, 0, 0, 0, 0)),
    ];
    let over_g = Expr::pow(Expr::var("g"), Expr::c(-1.0));
    let pendulum = Expr::Sqrt(Box::new(Expr::mul(vec![Expr::var("l"), over_g])));
    println!("\nformula checking");
    println!("  sqrt(l/g) has dimension {}", dimensional_check_formula(&pendulum, &vars).unwrap());
    assert_eq!(dimensional_check_formula(&pendulum, &vars).unwrap(), Dim::TIME);

    // A transcendental needs a pure number, because its series adds x to
    // x³. sin(omega*t) is meaningful; sin(t) is a missing timescale.
    let good = Expr::Sin(Box::new(Expr::mul(vec![Expr::var("omega"), Expr::var("t")])));
    let bad = Expr::Sin(Box::new(Expr::var("t")));
    println!("  sin(omega*t) checks out; sin(t) does not");
    assert!(dimensional_check_formula(&good, &vars).is_ok());
    assert!(dimensional_check_formula(&bad, &vars).is_err());

    // ---- 3. Buckingham's theorem, exactly ----------------------------
    //
    // Pipe flow: density, speed, diameter, viscosity. Four quantities,
    // three independent dimensions, so exactly one dimensionless group.
    let pipe = [
        Dim::new(-3, 1, 0, 0, 0, 0, 0),  // density   kg/m^3
        Dim::new(1, 0, -1, 0, 0, 0, 0),  // speed     m/s
        Dim::LENGTH,                     // diameter  m
        Dim::new(-1, 1, -1, 0, 0, 0, 0), // viscosity Pa s
    ];
    let groups = buckingham_pi(&pipe).expect("a well-posed problem");
    println!("\nBuckingham's theorem");
    println!("  4 quantities, rank 3 -> {} group", groups.len());
    let exponents: Vec<String> = groups[0].iter().map(|r| format!("{r}")).collect();
    println!("  exponents (rho, u, d, mu): {}", exponents.join(", "));
    println!("  that is rho^-1 u^-1 d^-1 mu, which is 1/Re -- the theorem finds");
    println!("  a basis for the null space, not the name anybody gave it");
    assert_eq!(groups.len(), 1);

    // The computation runs over exact rationals, not floats, because a
    // group is exactly in the null space or it is not -- and one that
    // cancelled to 1e-16 would be a rounding error reported as physics.
    println!("  computed over Rational, so the cancellation is exact, not 1e-16");
    assert!(groups[0].iter().all(|r| *r == Rational::from_i64(-1, 1) || *r == Rational::one()));

    // ---- 4. arithmetic without rounding ------------------------------
    let tenth = Rational::from_i64(1, 10);
    let fifth = Rational::from_i64(1, 5);
    let sum = tenth.add(&fifth);
    println!("\nexact arithmetic");
    println!("  0.1 + 0.2 in f64  = {:.17}", 0.1 + 0.2);
    println!("  1/10 + 1/5 exact  = {sum}");
    assert_ne!(0.1 + 0.2, 0.3);
    assert_eq!(sum, Rational::from_i64(3, 10));

    // An f64 is a dyadic rational, and from_f64_exact gives the value it
    // genuinely holds rather than the decimal it is printed as.
    let as_stored = Rational::from_f64_exact(0.1).expect("finite");
    println!("  and 0.1 as an f64 is really {as_stored}");
    assert_ne!(as_stored, Rational::from_i64(1, 10));
}
