//! Kani proof harnesses. Compiled only under `cargo kani` (`#[cfg(kani)]`
//! at the inclusion site in `lib.rs`).
//!
//! # Which harnesses run by default
//!
//! Seven of the twenty are behind the `kani-slow` feature and are off by
//! default. Every harness was timed individually against a five-minute
//! budget, and the result splits cleanly along what is being asserted:
//!
//! | asserts | outcome |
//! |---|---|
//! | panic-freedom, finiteness, a sign | verifies in 24-68 seconds |
//! | a numeric relation between symbolic float expressions | exceeds 300 seconds |
//!
//! CBMC decides floating point by bit-blasting it into a SAT instance.
//! Proving that a result is finite, or non-negative, or that a guard fires,
//! constrains few bits and lands quickly. Proving that one symbolic product
//! or quotient bounds another -- `p.contains(x * y)` for four corner
//! products, `orbital_velocity < escape_velocity` across two square roots,
//! `variance >= 0` over a summation -- constrains the whole 53-bit mantissa
//! of each intermediate, and the instance stops being tractable.
//!
//! That distinction is the one this module's physics harnesses already
//! describe: transcendentals are modelled as unconstrained finite values, so
//! those harnesses prove panic-freedom rather than numeric bounds. The slow
//! seven are the ones that ask for numeric bounds anyway.
//!
//! They are kept rather than deleted: each states something true and worth
//! stating, and a future Kani or solver may decide them. Run them with
//!
//! ```text
//! cargo kani --features kani-slow --harness <name>
//! ```
//!
//! Measured times, five-minute budget, one harness at a time:
//!
//! | harness | time |
//! |---|---|
//! | `normalize_angle_never_panics` | 24s |
//! | `displacement_is_finite_on_bounded_inputs` | 25s |
//! | `mat3_identity_inverse_is_identity` | 25s |
//! | `mean_panics_on_empty` | 25s |
//! | `projectile_range_panics_on_nonpositive_g` | 25s |
//! | `vec3_dot_with_self_nonnegative` | 25s |
//! | `kinetic_energy_nonnegative_for_nonnegative_mass` | 27s |
//! | `factorial_is_monotone_and_finite_below_171` | 28s |
//! | `projectile_range_never_panics_with_positive_g` | 28s |
//! | `escape_velocity_finite_nonnegative` | 32s |
//! | `shannon_entropy_never_panics_on_nonempty` | 34s |
//! | `vec3_normalized_never_produces_nan` | 52s |
//! | `lu_decompose_3x3_finite_or_err` | 68s |
//! | `bisection_result_is_inside_bracket` | over 300s |
//! | `interval_mul_contains_corner_products` | over 300s |
//! | `mat3_inverse_never_divides_by_zero` | over 300s |
//! | `mean_of_bounded_slice_is_bounded` | over 300s |
//! | `orbital_velocity_below_escape_velocity` | over 300s |
//! | `ray_aabb_interval_ordered` | over 300s |
//! | `variance_is_nonnegative` | over 300s |

pub mod core;
pub mod linalg;
pub mod physics;
pub mod spatial;
