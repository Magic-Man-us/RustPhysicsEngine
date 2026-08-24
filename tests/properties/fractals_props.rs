//! Properties for `fractals`: L-system growth laws and IFS attractor
//! invariants.

use rust_physics_engine::fractals::ifs::presets as ifs_presets;
use rust_physics_engine::fractals::lsystem::{presets, Turtle2};
use rust_physics_engine::monte_carlo::Rng;

#[test]
fn prop_lsystem_segment_growth_laws() {
    // Segment counts follow each system's branching factor exactly.
    let cases: [(&str, rust_physics_engine::fractals::lsystem::LSystem, usize); 4] = [
        ("koch", presets::koch_curve(), 4),
        ("dragon", presets::dragon(), 2),
        ("arrowhead", presets::sierpinski_arrowhead(), 3),
        ("gosper", presets::gosper(), 7),
    ];
    for (name, ls, factor) in cases {
        let mut prev = 0usize;
        for n in 1..=4 {
            let s = ls.generate(n, None);
            let mut t = Turtle2::new(1.0, ls.angle.to_degrees());
            let count = t.interpret(&s).len();
            if n > 1 {
                assert_eq!(count, prev * factor, "{name} multiplies segments by {factor}");
            }
            prev = count;
        }
    }
}

#[test]
fn prop_moran_dimensions_match_closed_forms() {
    for (name, ifs, expected) in [
        ("sierpinski", ifs_presets::sierpinski(), 3.0f64.ln() / 2.0f64.ln()),
        ("koch", ifs_presets::koch(), 4.0f64.ln() / 3.0f64.ln()),
        ("carpet", ifs_presets::sierpinski_carpet(), 8.0f64.ln() / 3.0f64.ln()),
        ("vicsek", ifs_presets::vicsek(), 5.0f64.ln() / 3.0f64.ln()),
        ("hexaflake", ifs_presets::hexaflake(), 7.0f64.ln() / 3.0f64.ln()),
        ("cantor_dust", ifs_presets::cantor_dust(), 4.0f64.ln() / 3.0f64.ln()),
        ("levy", ifs_presets::levy(), 2.0),
        ("dragon", ifs_presets::dragon(), 2.0),
    ] {
        let d = ifs.similarity_dimension().unwrap_or_else(|| panic!("{name} is a similitude IFS"));
        assert!((d - expected).abs() < 1e-9, "{name} dimension {d} vs {expected}");
    }
}

#[test]
fn prop_chaos_game_reproducible_and_attractor_invariant() {
    // Same seed, same trajectory.
    let ifs = ifs_presets::barnsley_fern();
    let a = ifs.chaos_game(500, 20, &mut Rng::new(99));
    let b = ifs.chaos_game(500, 20, &mut Rng::new(99));
    assert_eq!(a.len(), b.len());
    for (p, q) in a.iter().zip(&b) {
        assert_eq!(p, q, "chaos game is deterministic per seed");
    }
    // The attractor is invariant: applying any map to an attractor
    // point lands on (near) the attractor.
    let samples = ifs.chaos_game(50_000, 20, &mut Rng::new(7));
    for (i, &p) in samples.iter().enumerate().step_by(5000) {
        let (map, _) = &ifs.maps[i % ifs.maps.len()];
        let q = map.apply(p);
        let d = samples
            .iter()
            .map(|s| s.distance_to(&q))
            .fold(f64::INFINITY, f64::min);
        assert!(d < 0.05, "image of attractor point stays on the attractor ({d})");
    }
}

#[test]
fn prop_mandelbrot_interior_regions_never_escape() {
    use rust_physics_engine::fractals::escape_time::{
        mandelbrot, mandelbrot_in_main_cardioid, mandelbrot_in_period2_bulb, EscapeParams,
    };
    use rust_physics_engine::fractals::Complex;
    let params = EscapeParams { max_iter: 10_000, ..EscapeParams::default() };
    let mut rng = Rng::new(41);
    let mut tested = 0;
    while tested < 40 {
        let c = Complex::new(rng.next_f64() * 3.0 - 2.25, rng.next_f64() * 2.5 - 1.25);
        if mandelbrot_in_main_cardioid(c) || mandelbrot_in_period2_bulb(c) {
            assert!(
                !mandelbrot(c, &params).escaped,
                "known interior point ({}, {}) escaped",
                c.re,
                c.im
            );
            tested += 1;
        }
    }
}

#[test]
fn prop_noise_seeds_uncorrelated() {
    use rust_physics_engine::fractals::noise::{OpenSimplex2, Perlin};
    let pa = Perlin::new(100);
    let pb = Perlin::new(200);
    let sa = OpenSimplex2::new(100);
    let sb = OpenSimplex2::new(200);
    let mut rng = Rng::new(77);
    let n = 20_000;
    let (mut sum_a, mut sum_b, mut sum_ab, mut sum_a2, mut sum_b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut t_a, mut t_b, mut t_ab, mut t_a2, mut t_b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for _ in 0..n {
        let (x, y) = (rng.next_f64() * 100.0, rng.next_f64() * 100.0);
        let (a, b) = (pa.noise_2d(x, y), pb.noise_2d(x, y));
        sum_a += a;
        sum_b += b;
        sum_ab += a * b;
        sum_a2 += a * a;
        sum_b2 += b * b;
        let (u, v) = (sa.noise_2d(x, y), sb.noise_2d(x, y));
        t_a += u;
        t_b += v;
        t_ab += u * v;
        t_a2 += u * u;
        t_b2 += v * v;
    }
    let nf = n as f64;
    let r_perlin = (sum_ab / nf - sum_a / nf * (sum_b / nf))
        / ((sum_a2 / nf - (sum_a / nf).powi(2)).sqrt()
            * (sum_b2 / nf - (sum_b / nf).powi(2)).sqrt());
    assert!(r_perlin.abs() < 0.05, "Perlin seeds correlated (r = {r_perlin})");
    let r_simplex = (t_ab / nf - t_a / nf * (t_b / nf))
        / ((t_a2 / nf - (t_a / nf).powi(2)).sqrt() * (t_b2 / nf - (t_b / nf).powi(2)).sqrt());
    assert!(r_simplex.abs() < 0.05, "simplex seeds correlated (r = {r_simplex})");
}

#[test]
fn prop_julia_c_zero_unit_circle() {
    use rust_physics_engine::fractals::escape_time::{julia, EscapeParams};
    use rust_physics_engine::fractals::Complex;
    let params = EscapeParams { max_iter: 3000, ..EscapeParams::default() };
    let mut rng = Rng::new(55);
    for _ in 0..60 {
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let inner = 0.999 * rng.next_f64();
        let outer = 1.001 + rng.next_f64();
        let zi = Complex::new(inner * angle.cos(), inner * angle.sin());
        let zo = Complex::new(outer * angle.cos(), outer * angle.sin());
        let c = Complex::new(0.0, 0.0);
        assert!(!julia(zi, c, &params).escaped, "|z| < 1 bounded");
        assert!(julia(zo, c, &params).escaped, "|z| > 1 escapes");
    }
}

#[test]
fn prop_rule90_has_sierpinski_dimension() {
    use rust_physics_engine::fractals::automata::Ca1D;
    use rust_physics_engine::fractals::box_count_2d;
    // Rule 90 from a single seed draws the Sierpinski gasket; its
    // box-count slope matches log 3 / log 2.
    let mut ca = Ca1D::new(90, 257, false);
    ca.seed_center();
    let rows = ca.run(128);
    let points: Vec<(f64, f64)> = rows
        .iter()
        .enumerate()
        .flat_map(|(y, row)| {
            row.iter()
                .enumerate()
                .filter(|&(_, &alive)| alive)
                .map(move |(x, _)| (x as f64, y as f64))
        })
        .collect();
    let bounds = (0.0, 257.0, 0.0, 129.0);
    let n1 = box_count_2d(&points, 16, bounds) as f64;
    let n2 = box_count_2d(&points, 64, bounds) as f64;
    let dim = (n2 / n1).ln() / 4.0f64.ln();
    let expected = 3.0f64.ln() / 2.0f64.ln();
    assert!((dim - expected).abs() < 0.12, "rule 90 dimension {dim} vs {expected}");
}

#[test]
fn prop_lyapunov_signs_distinguish_chaos() {
    use rust_physics_engine::fractals::attractors::{presets, Attractor3};
    use rust_physics_engine::math::Vec3;
    // Chaotic flows have a positive largest exponent; a damped
    // linear oscillator has all exponents negative.
    let lorenz = presets::lorenz(10.0, 28.0, 8.0 / 3.0);
    let l = lorenz.lyapunov_spectrum(Vec3::new(1.0, 1.0, 1.0), 20_000, 0.005);
    assert!(l[0] > 0.5, "Lorenz is chaotic ({})", l[0]);
    let damped = Attractor3 {
        derivs: Box::new(|p: Vec3| Vec3::new(p.y, -p.x - 0.5 * p.y, -p.z)),
        dt: 0.01,
    };
    let d = damped.lyapunov_spectrum(Vec3::new(1.0, 0.0, 1.0), 20_000, 0.01);
    assert!(d[0] < 0.0, "damped flow contracts ({})", d[0]);
    // Rossler: weakly chaotic, positive but small.
    let rossler = presets::rossler(0.2, 0.2, 5.7);
    let r = rossler.lyapunov_spectrum(Vec3::new(1.0, 1.0, 1.0), 40_000, 0.02);
    assert!(r[0] > 0.02 && r[0] < 0.2, "Rossler exponent {}", r[0]);
}

#[test]
fn prop_gray_scott_regimes_bounded_and_deterministic() {
    use rust_physics_engine::fractals::automata::GrayScott;
    for make in [
        GrayScott::mitosis as fn(usize, usize) -> GrayScott,
        GrayScott::coral,
        GrayScott::worms,
        GrayScott::maze,
    ] {
        let mut a = make(24, 24);
        a.seed_square(10, 10, 4);
        a.run(120);
        for (&u, &v) in a.u.iter().zip(&a.v) {
            assert!((0.0..=1.5).contains(&u) && (0.0..=1.5).contains(&v));
        }
        let mut b = make(24, 24);
        b.seed_square(10, 10, 4);
        b.run(120);
        assert_eq!(a.u, b.u, "deterministic evolution");
    }
}
