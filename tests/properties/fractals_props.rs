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
