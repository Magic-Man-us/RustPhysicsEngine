//! Properties of the units and dimensional analysis module.
//!
//! Almost everything here is an identity over integers or exact
//! rationals, so almost nothing needs a tolerance.
//!
//! *Exponent arithmetic.* Multiplying quantities adds their exponent
//! vectors and dividing subtracts them, so `(a * b) / b` has exactly
//! `a`'s dimension -- asserted with `==` on seven signed bytes, not with
//! a comparison of floats. Raising to a power multiplies every exponent,
//! and a square root exists exactly when every one of them is even.
//!
//! *Buckingham's theorem is a rank computation.* The number of
//! dimensionless groups is exactly the quantity count minus the rank of
//! the dimension matrix, and each group's exponents cancel every
//! dimension exactly, checked in [`Rational`] arithmetic. A group that
//! cancelled to `1e-16` would be a rounding error reported as physics,
//! and in floating point there would be no way to tell the two apart.
//!
//! *Conversions compose.* Converting from one unit to another and back
//! returns the value; converting through a third gives the same answer
//! as going directly. Converting between different dimensions is an
//! error, which is the point of the exercise rather than a detail.

use rust_physics_engine::exact::rational::Rational;
use rust_physics_engine::units::dimensional::{
    buckingham_pi, is_dimensionless_group, natural_units_convert, natural_units_power,
};
use rust_physics_engine::units::quantity::{
    parse_quantity, parse_unit, si_prefixes_format, unit_convert, Dim, DimError, Quantity,
};
use rust_physics_engine::monte_carlo::Rng;

/// A random dimension with small exponents, so that products stay
/// inside a signed byte.
fn dim(rng: &mut Rng) -> Dim {
    let mut e = [0i8; 7];
    for slot in e.iter_mut() {
        *slot = (rng.below(7) as i8) - 3;
    }
    Dim::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6])
}

#[test]
fn prop_multiplying_adds_exponents_and_dividing_takes_them_back() {
    let mut rng = Rng::new(0x4a91_02cd);
    for _ in 0..80 {
        let a = dim(&mut rng);
        let b = dim(&mut rng);
        let product = a.mul(&b).unwrap();
        // Every exponent is the sum, exactly.
        for k in 0..7 {
            assert_eq!(product.exponents()[k], a.exponents()[k] + b.exponents()[k]);
        }
        // And dividing takes it straight back.
        assert_eq!(product.div(&b).unwrap(), a);
        assert_eq!(product.div(&a).unwrap(), b);
        // Multiplication commutes and the dimensionless vector is its
        // identity.
        assert_eq!(a.mul(&b).unwrap(), b.mul(&a).unwrap());
        assert_eq!(a.mul(&Dim::NONE).unwrap(), a);
        assert_eq!(a.div(&a).unwrap(), Dim::NONE);
        assert!(a.div(&a).unwrap().is_dimensionless());
        // A power multiplies every exponent, and squaring is
        // multiplying by itself.
        assert_eq!(a.pow(1).unwrap(), a);
        assert_eq!(a.pow(0).unwrap(), Dim::NONE);
        assert_eq!(a.pow(2).unwrap(), a.mul(&a).unwrap());
        assert_eq!(a.pow(-1).unwrap(), Dim::NONE.div(&a).unwrap());
        // The square of anything has an exact root, and it is the
        // original.
        assert_eq!(a.pow(2).unwrap().sqrt().unwrap(), a);
        // An odd exponent anywhere means no root at all.
        let odd = a.exponents().iter().any(|e| e % 2 != 0);
        assert_eq!(a.sqrt().is_err(), odd);
    }
}

#[test]
fn prop_quantities_carry_their_dimensions_through_arithmetic() {
    let mut rng = Rng::new(0x18c0_7fe4);
    for _ in 0..60 {
        let da = dim(&mut rng);
        let db = dim(&mut rng);
        let a = Quantity::new(1.0 + 9.0 * rng.next_f64(), da);
        let b = Quantity::new(1.0 + 9.0 * rng.next_f64(), db);
        // The dimension of a product is the product of the dimensions,
        // and the value is the product of the values.
        let p = a.mul(&b).unwrap();
        assert_eq!(p.dim, da.mul(&db).unwrap());
        assert!((p.value - a.value * b.value).abs() < 1e-12 * p.value.abs());
        // Dividing by b recovers a exactly in dimension and to rounding
        // in value.
        let back = p.div(&b).unwrap();
        assert_eq!(back.dim, da);
        assert!((back.value - a.value).abs() < 1e-12 * a.value.abs());
        // Adding is allowed exactly when the dimensions agree.
        assert_eq!(a.add(&b).is_ok(), da == db);
        assert_eq!(a.sub(&b).is_ok(), da == db);
        if da != db {
            assert_eq!(
                a.add(&b),
                Err(DimError::Mismatch { expected: da, found: db })
            );
        }
        // Adding a quantity to itself doubles it and keeps the
        // dimension.
        let doubled = a.add(&a).unwrap();
        assert_eq!(doubled.dim, da);
        assert!((doubled.value - 2.0 * a.value).abs() < 1e-12 * a.value.abs());
        assert!(a.sub(&a).unwrap().value.abs() < 1e-12 * a.value.abs());
        // Squaring then rooting is the identity on both parts.
        let squared = a.pow(2).unwrap();
        let rooted = squared.sqrt().unwrap();
        assert_eq!(rooted.dim, da);
        assert!((rooted.value - a.value).abs() < 1e-9 * a.value.abs());
    }
}

#[test]
fn prop_conversions_round_trip_and_compose() {
    let mut rng = Rng::new(0x77e2_5a10);
    let families: [&[&str]; 5] = [
        &["m", "km", "cm", "mm", "ft", "in", "mi"],
        &["kg", "g", "mg", "lb", "t"],
        &["s", "ms", "min", "h", "d", "yr"],
        &["J", "kJ", "eV", "Wh", "kWh", "cal"],
        &["Pa", "kPa", "bar", "atm"],
    ];
    for _ in 0..60 {
        let family = families[rng.below(families.len() as u64) as usize];
        let a = family[rng.below(family.len() as u64) as usize];
        let b = family[rng.below(family.len() as u64) as usize];
        let c = family[rng.below(family.len() as u64) as usize];
        let value = 0.5 + 100.0 * rng.next_f64();
        // There and back.
        let there = unit_convert(value, a, b).unwrap();
        let back = unit_convert(there, b, a).unwrap();
        assert!((back - value).abs() < 1e-9 * value, "{a} to {b} and back gave {back}");
        // Composing two conversions equals doing it directly, which is
        // the statement that every unit in a family shares one scale.
        let direct = unit_convert(value, a, c).unwrap();
        let indirect = unit_convert(unit_convert(value, a, b).unwrap(), b, c).unwrap();
        assert!(
            (direct - indirect).abs() < 1e-9 * direct.abs().max(1.0),
            "{a}->{c} directly gave {direct}, via {b} gave {indirect}"
        );
        // Converting to itself changes nothing.
        assert!((unit_convert(value, a, a).unwrap() - value).abs() < 1e-12 * value);
        // Conversion is linear in the value.
        let scaled = unit_convert(3.0 * value, a, b).unwrap();
        assert!((scaled - 3.0 * there).abs() < 1e-9 * scaled.abs().max(1.0));
    }
    // Across families it is always an error, which is the whole point.
    for _ in 0..30 {
        let i = rng.below(families.len() as u64) as usize;
        let mut j = rng.below(families.len() as u64) as usize;
        if i == j {
            j = (j + 1) % families.len();
        }
        let a = families[i][rng.below(families[i].len() as u64) as usize];
        let b = families[j][rng.below(families[j].len() as u64) as usize];
        assert!(
            matches!(unit_convert(1.0, a, b), Err(DimError::Mismatch { .. })),
            "{a} to {b} was allowed"
        );
    }
}

#[test]
fn prop_parsing_and_conversion_agree() {
    // Parsing "<v> <unit>" must give the same SI magnitude as converting
    // v from that unit, since they are two routes through the same
    // table.
    let mut rng = Rng::new(0x2b40_9dd1);
    let units = [
        "m", "km", "mm", "ft", "mi", "kg", "g", "lb", "s", "min", "h", "J", "eV", "kWh",
        "Pa", "bar", "N", "W", "V", "A", "K", "Hz", "L",
    ];
    for _ in 0..80 {
        let unit = units[rng.below(units.len() as u64) as usize];
        let value = 0.25 + 50.0 * rng.next_f64();
        let parsed = parse_quantity(&format!("{value} {unit}")).unwrap();
        let (factor, dimension) = parse_unit(unit).unwrap();
        assert_eq!(parsed.dim, dimension);
        assert!(
            (parsed.value - value * factor).abs() < 1e-9 * parsed.value.abs().max(1e-30),
            "{value} {unit} parsed to {}",
            parsed.value
        );
        // And reading it back out in the same unit returns the number.
        let out = parsed.to(unit).unwrap();
        assert!((out - value).abs() < 1e-9 * value, "{value} {unit} came back as {out}");
        // Reading it out in a unit of another dimension is refused.
        assert!(parsed.to("mol").is_err() || dimension == Dim::AMOUNT);
    }
}

#[test]
fn prop_the_prefix_formatter_preserves_the_value() {
    // The scaled number times its prefix's power of ten is the original,
    // and the scaled number lies in [1, 1000) wherever a prefix exists.
    let mut rng = Rng::new(0x5d13_ba82);
    let power = |p: &str| -> f64 {
        match p {
            "Y" => 1e24, "Z" => 1e21, "E" => 1e18, "P" => 1e15, "T" => 1e12,
            "G" => 1e9, "M" => 1e6, "k" => 1e3, "" => 1.0, "m" => 1e-3,
            "u" => 1e-6, "n" => 1e-9, "p" => 1e-12, "f" => 1e-15, "a" => 1e-18,
            "z" => 1e-21, "y" => 1e-24,
            _ => panic!("unexpected prefix {p}"),
        }
    };
    for _ in 0..100 {
        let exponent = (rng.below(41) as i32) - 20;
        let mantissa = 1.0 + 8.0 * rng.next_f64();
        let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        let value = sign * mantissa * 10f64.powi(exponent);
        let (scaled, prefix) = si_prefixes_format(value);
        let rebuilt = scaled * power(prefix);
        assert!(
            (rebuilt - value).abs() < 1e-9 * value.abs(),
            "{value} became {scaled}{prefix}"
        );
        assert!(scaled.abs() >= 1.0 && scaled.abs() < 1000.0, "{value} scaled to {scaled}");
        // The sign survives.
        assert_eq!(scaled < 0.0, value < 0.0);
    }
}

#[test]
fn prop_buckingham_returns_exactly_the_null_space() {
    let mut rng = Rng::new(0x3e07_45ab);
    for _ in 0..40 {
        let n = 2 + (rng.below(6)) as usize;
        let dims: Vec<Dim> = (0..n).map(|_| dim(&mut rng)).collect();
        let groups = buckingham_pi(&dims).unwrap();
        // Every returned group really is dimensionless, exactly.
        for g in &groups {
            assert_eq!(g.len(), n);
            assert!(
                is_dimensionless_group(&dims, g).unwrap(),
                "a returned group did not cancel"
            );
        }
        // The count is the quantity count minus the rank, and the rank
        // is at most seven because there are seven base dimensions.
        let rank = n - groups.len();
        assert!(rank <= 7.min(n), "the rank came out at {rank}");
        // Any linear combination of the basis is also dimensionless,
        // which is what makes it a basis of a subspace rather than a
        // list of coincidences.
        if groups.len() >= 2 {
            let alpha = Rational::from_i64(1 + rng.below(5) as i64, 1 + rng.below(3) as i64);
            let beta = Rational::from_i64(-(1 + rng.below(4) as i64), 1 + rng.below(3) as i64);
            let mixed: Vec<Rational> = (0..n)
                .map(|k| alpha.mul(&groups[0][k]).add(&beta.mul(&groups[1][k])))
                .collect();
            assert!(is_dimensionless_group(&dims, &mixed).unwrap());
        }
        // The basis is independent: no group is entirely zero, since
        // each carries a one in its own free column.
        for g in &groups {
            assert!(g.iter().any(|r| !r.is_zero()), "a group was the zero vector");
        }
    }
}

#[test]
fn prop_buckingham_agrees_with_the_rank_it_implies() {
    // Building a problem whose rank is known in advance: r independent
    // base dimensions plus k quantities made only from them must give
    // exactly k groups.
    let mut rng = Rng::new(0x6cb2_0f39);
    let bases = [Dim::LENGTH, Dim::MASS, Dim::TIME, Dim::CURRENT];
    for _ in 0..30 {
        let r = 1 + (rng.below(4)) as usize;
        let extra = 1 + (rng.below(4)) as usize;
        let mut dims: Vec<Dim> = bases[..r].to_vec();
        for _ in 0..extra {
            // A product of powers of the chosen bases, so it adds
            // nothing to the rank.
            let mut d = Dim::NONE;
            for base in &bases[..r] {
                let p = (rng.below(5) as i8) - 2;
                d = d.mul(&base.pow(p).unwrap()).unwrap();
            }
            dims.push(d);
        }
        let groups = buckingham_pi(&dims).unwrap();
        assert_eq!(
            groups.len(),
            extra,
            "{r} bases and {extra} derived quantities gave {} groups",
            groups.len()
        );
        for g in &groups {
            assert!(is_dimensionless_group(&dims, g).unwrap());
        }
    }
}

#[test]
fn prop_natural_units_are_a_consistent_change_of_bookkeeping() {
    let mut rng = Rng::new(0x0a75_c3e6);
    for _ in 0..60 {
        let mut d = dim(&mut rng);
        // Only the mechanical dimensions have a natural-unit power.
        d = Dim::new(d.m, d.kg, d.s, 0, 0, 0, 0);
        let power = natural_units_power(d).unwrap();
        assert_eq!(power, d.kg as i32 - d.m as i32 - d.s as i32);
        // Multiplying two quantities adds their powers, which is what
        // makes the bookkeeping consistent rather than a coincidence of
        // the three factors.
        let e = {
            let f = dim(&mut rng);
            Dim::new(f.m, f.kg, f.s, 0, 0, 0, 0)
        };
        if let Ok(product) = d.mul(&e) {
            assert_eq!(
                natural_units_power(product).unwrap(),
                power + natural_units_power(e).unwrap()
            );
            // And the converted magnitudes multiply too.
            let (x, y) = (1.0 + rng.next_f64(), 1.0 + rng.next_f64());
            let left = natural_units_convert(x, d).unwrap() * natural_units_convert(y, e).unwrap();
            let right = natural_units_convert(x * y, product).unwrap();
            assert!(
                (left - right).abs() < 1e-9 * left.abs().max(1e-300),
                "the conversion did not multiply: {left} against {right}"
            );
        }
        // Anything electromagnetic or thermal is refused rather than
        // guessed at.
        let charged = Dim::new(d.m, d.kg, d.s, 1, 0, 0, 0);
        assert!(natural_units_power(charged).is_err());
        assert!(natural_units_convert(1.0, charged).is_err());
    }
}
