# Guide

A walk through the library by doing things with it, rather than a list of
what it contains. For that list see [the module map](MODULE_MAP.md); for
the API reference run `cargo doc --no-deps --open`.

Every code block below is a real file in [`examples/`](../examples), compiled
and run by CI. The output shown is what it actually prints. If a chapter
here describes something that does not work, CI fails.

```bash
cargo run --example guide_02_orbit
cargo run --example guide_03_signal
cargo run --example guide_04_fem
cargo run --example guide_05_correctness
```

**Contents**

1. [Getting your bearings](#1-getting-your-bearings)
2. [A spacecraft in orbit](#2-a-spacecraft-in-orbit)
3. [A tone buried in noise](#3-a-tone-buried-in-noise)
4. [Solving a differential equation](#4-solving-a-differential-equation)
5. [The tools for not being wrong](#5-the-tools-for-not-being-wrong)
6. [Where to look for things](#6-where-to-look-for-things)
7. [Things that will bite you](#7-things-that-will-bite-you)

---

## 1. Getting your bearings

Add it to `Cargo.toml`:

```toml
[dependencies]
rust_physics_engine = { git = "https://github.com/Magic-Man-us/RustPhysicsEngine" }
```

There is nothing else to install. The crate has no dependencies — `Cargo.lock`
holds exactly one package, itself — so there is no feature matrix to learn and
no transitive tree to audit.

Three conventions hold everywhere:

- **SI units, angles in radians.** A function that wants something else says
  so in its documentation.
- **`f64`**, except where exactness is the point. [`exact`] works over
  arbitrary-precision integers and rationals.
- **Solvers return `Result`.** A solver that cannot converge tells you, rather
  than returning a number that looks like an answer.

The smallest useful thing:

```rust
use rust_physics_engine::classical::projectile_range;

// v₀ = 50 m/s, θ = 45°, g = 9.81 m/s²
let range = projectile_range(50.0, std::f64::consts::FRAC_PI_4, 9.81);
// 254.84 m
```

---

## 2. A spacecraft in orbit

📄 [`examples/guide_02_orbit.rs`](../examples/guide_02_orbit.rs)

Orbital mechanics never uses `G` and `M` separately — only their product, the
gravitational parameter μ. That is the first thing to internalise, because
every formula in [`astrophysics`] takes μ.

```rust
let mu = G * EARTH_MASS;

// A circular orbit 400 km up, near enough the ISS.
let r = EARTH_RADIUS + 400e3;
let speed = (mu / r).sqrt();
let position = Vec3::new(r, 0.0, 0.0);
let velocity = Vec3::new(0.0, speed, 0.0);

let elements = OrbitalElements::from_state_vectors(position, velocity, mu);
```

Going from a state vector to Keplerian elements is the first thing you do with
tracking data, because elements are what you can reason about — a position and
velocity tell you where something is, elements tell you what it is doing.

```
circular orbit at 400 km
  speed          7673 m/s
  semi-major     6771.0 km
  eccentricity   0.00e0
  period         92.4 min
  bound?         true
```

Those are the real ISS numbers. A circular orbit has `e = 0` to within
rounding, and its period is Kepler's third law — both worth asserting rather
than eyeballing:

```rust
assert!(elements.eccentricity < 1e-12);
let kepler = 2.0 * PI * (r.powi(3) / mu).sqrt();
assert!((elements.period(mu) - kepler).abs() < 1e-6);
```

### Getting somewhere else

A Hohmann transfer is two burns: one to enter an ellipse that touches both
circles, one to circularise at the far end.

```rust
let (dv1, dv2) = hohmann_delta_v(mu, r, 42_164e3);  // to geostationary
```

```
Hohmann transfer to geostationary
  burn 1         2399 m/s
  burn 2         1457 m/s
  total          3857 m/s
  flight time    5.3 hours
```

The transfer ellipse touches both circles, so its semi-major axis is the mean
of the two radii, and half its period is the flight time. You can check the
whole thing closes with vis-viva, `v² = μ(2/r − 1/a)`:

```rust
let a_transfer = 0.5 * (r + r_geo);
let v_peri = (mu * (2.0 / r - 1.0 / a_transfer)).sqrt();
assert!((v_peri - (speed + dv1)).abs() < 1e-6);   // burn 1 lands you on it
```

**Where to go next.** [`astrophysics::kepler`] solves Kepler's equation for
elliptic, parabolic and hyperbolic orbits including e = 0.99;
[`astrophysics::lambert`] finds the transfer connecting two positions in a
given time; [`astrophysics::maneuvers`] covers plane changes, phasing and J2
drift; [`astrophysics::nbody`] integrates many bodies at once.

---

## 3. A tone buried in noise

📄 [`examples/guide_03_signal.rs`](../examples/guide_03_signal.rs)

Two tones and noise at more than the amplitude of the signal. We want the
440 Hz one and not the 2.6 kHz one.

```rust
let spectrum = rfft(&signal);   // real input -> non-negative frequencies only
```

`rfft` returns only the non-negative frequencies, which is all a real signal
has: bin `k` sits at `k·fs/n` Hz.

```
before filtering
  tone found at      439 Hz  (magnitude 1797)
  interference at    2600 Hz  (magnitude 1556)
```

Both tones come out clearly despite the noise, because noise spreads across
every bin while a sinusoid concentrates into one. That is the whole reason
the FFT is the first tool you reach for.

### Filtering

```rust
// Cutoff is in cycles per sample, so 1 kHz at fs = 8 kHz is 0.125.
let taps = fir_lowpass(101, 1_000.0 / fs, WindowKind::Hamming);
let filtered = fir_apply(&taps, &signal);
```

```
after a 1 kHz low-pass
  440 Hz kept        magnitude 1782
  2.6 kHz rejected   magnitude 1
  rejection          67 dB
```

More taps means a sharper transition between passband and stopband, at the
cost of more delay and more arithmetic. 101 taps buys 67 dB here.

### Getting the noise floor instead of the peak

Welch's method averages periodograms over overlapping segments. It trades
frequency resolution for a reduction in variance, which is what you want when
you care about the noise floor rather than the exact peak:

```rust
let (freqs, psd) = welch(&signal, fs, 512, 256, WindowKind::Hann);
```

**Where to go next.** [`transforms::fft`] handles any length, not just powers
of two — Bluestein's chirp-z covers the prime ones. [`dsp::iir`] has RBJ
biquads and second-order-section cascades when you want a filter that is cheap
rather than linear-phase. [`transforms::wavelet`] is the tool when the
frequency content changes over time. [`transforms::spectral`] adds multitaper
and Lomb–Scargle, the latter for unevenly sampled data.

---

## 4. Solving a differential equation

📄 [`examples/guide_04_fem.rs`](../examples/guide_04_fem.rs)

Solve `−u″ = f` on `[0, 1]` with `u(0) = u(1) = 0`. Choosing
`f = π² sin(πx)` makes the exact answer `u = sin(πx)`, which is what makes the
error *measurable* rather than merely plausible.

```rust
let values = fem_1d_poisson(&f, 0.0, 1.0, (Bc::Dirichlet(0.0), Bc::Dirichlet(0.0)), n)?;
let solution = Fem1dSolution::new(0.0, 1.0, 1, values)?;

let e_l2 = fem_1d_error_l2(&solution, &exact);
let e_h1 = fem_1d_error_h1_seminorm(&solution, &d_exact);
```

```
  P1 elements
   cells       h      L2 error      H1 error
       8  0.1250      9.921e-3      2.512e-1
      16  0.0625      2.487e-3      1.258e-1
      32  0.0312      6.220e-4      6.295e-2
      64  0.0156      1.555e-4      3.148e-2
     128  0.0078      3.888e-5      1.574e-2

  measured rate   L2 2.00   H1 1.00
  theory          L2 2.00   H1 1.00
```

**This is the part worth pausing on.** An error that merely shrinks tells you
nothing — almost any wrong method produces a shrinking error. An error that
shrinks at *exactly* `h²` tells you the discretisation is the one you think it
is. The rate is the slope of `log(error)` against `log(h)`, and it is
predicted before it is measured.

The H1 rate is one lower than L2 because the energy norm measures the
derivative, and differentiating a piecewise polynomial costs you an order.

Quadratic elements buy an order in each norm on the same mesh:

```
  P2 elements   measured rate L2 3.00   theory 3.00
```

**Why finite elements rather than finite differences.** A finite difference
replaces the derivative with a difference quotient and asks the equation to
hold at grid points. A finite element multiplies by a test function, integrates
by parts, and asks the resulting integral identity to hold across a
finite-dimensional space. That change buys two things: the method needs one
less derivative of the solution to make sense, so a kink in the coefficient is
admissible rather than fatal; and the answer is the *best* approximation in the
space under the energy norm — not close to the best, the best.

**Where to go next.** [`fem::fem2d`] does triangular elements in the plane:
Poisson, Helmholtz, drum eigenvalues, plane-stress elasticity, transient heat.
[`fem::fdtd`] is Maxwell on a Yee grid with PML. [`fem::spectral_pde`] trades
matrix sparsity for a convergence rate limited only by smoothness. [`cfd`] has
the fluid-specific schemes; [`sim`] has compact readable integrators when you
want to watch something move rather than converge.

---

## 5. The tools for not being wrong

📄 [`examples/guide_05_correctness.rs`](../examples/guide_05_correctness.rs)

The two most expensive unit mistakes on record — the Mars Climate Orbiter's
pound-seconds fed to a newton-second interface, and the Gimli Glider's
kilograms of fuel loaded as pounds — were both arithmetic a computer performed
correctly on numbers that meant something other than the receiving code
assumed. Neither would have been caught by testing the arithmetic.

### Dimensions in the type

```rust
let work = force.mul(&distance)?;    // exactly joules
force.add(&time)                     // Err: dimension mismatch
```

```
  4.45 N x 2 m = 8.90 m^2 kg s^-2
  adding a force to a time -> dimension mismatch: expected m kg s^-2, found s
  sqrt(9 m^2)  = 3.0 m
```

Multiplication adds the seven exponents, division subtracts them, and a square
root exists only when every one of them is even — there is no square root of a
metre, so that is a refusal rather than a rounding decision.

### Checking a formula rather than a number

Both sides of `x + v` are perfectly good floats, so no amount of *running* a
formula finds that mistake. Walking the expression does:

```rust
let pendulum = Expr::Sqrt(Box::new(Expr::mul(vec![Expr::var("l"), over_g])));
dimensional_check_formula(&pendulum, &vars)?   // -> s
```

```
  sqrt(l/g) has dimension s
  sin(omega*t) checks out; sin(t) does not
```

A transcendental's argument must be dimensionless, because its series adds `x`
to `x³`. `exp(−t/τ)` is meaningful and `exp(−t)` is not — and the difference is
a missing timescale, which is a real bug that produces finite numbers.

### Buckingham's theorem, exactly

```
  4 quantities, rank 3 -> 1 group
  exponents (rho, u, d, mu): -1, -1, -1, 1
  that is rho^-1 u^-1 d^-1 mu, which is 1/Re
```

The theorem says how many dimensionless groups there are — quantity count
minus the rank of the dimension matrix — not which ones. Any basis of the null
space works, and Reynolds is a particular choice made for physical reasons the
algebra knows nothing about.

The computation runs over exact `Rational`, not floats, and that is not
fastidiousness: a group is exactly in the null space or it is not, and one that
cancelled to `1e-16` would be a rounding error reported as physics. In floating
point there is no way to tell those apart.

### Arithmetic without rounding

```
  0.1 + 0.2 in f64  = 0.30000000000000004
  1/10 + 1/5 exact  = 3/10
  and 0.1 as an f64 is really 3602879701896397/36028797018963968
```

That last line is the useful one. `0.1` is not one tenth; it is a
power-of-two fraction near it. `Rational::from_f64_exact` gives you the value
the float genuinely holds rather than the decimal it is printed as.

---

## 6. Where to look for things

| If you want to… | Start at |
|---|---|
| throw, drop, collide, oscillate | [`classical`], [`resonance`] |
| bend or load a structure | [`solid_mechanics`], [`continuum_mechanics`] |
| move heat around | [`thermodynamics`], [`sim::heat_sim`] |
| do circuits or fields | [`electromagnetism`], [`electronics`], [`rf`] |
| filter or transform a signal | [`transforms`], [`dsp`], [`signal_processing`] |
| make or analyse sound | [`audio`], [`acoustics`] |
| move a fluid | [`fluids`] for relations, [`cfd`] for solvers |
| go to orbit | [`astrophysics`], [`propulsion`] |
| do quantum mechanics | [`quantum`] |
| solve a PDE properly | [`fem`] |
| fit or classify data | [`learn`], [`statistics`] |
| optimise something | [`optimization`] |
| price or hedge something | [`finance`] |
| work in more than 3 dimensions | [`manifold`] |
| index or intersect geometry | [`spatial`], [`mesh`] |
| avoid a unit mistake | [`units`] |
| avoid a rounding mistake | [`exact`], [`core`] |

The [module map](MODULE_MAP.md) has all 295 modules with sizes and summaries.

---

## 7. Things that will bite you

**`Rng` is a linear congruential generator that returns its raw state.** The
low bits have a short period, so `next_u64() % m` for a power-of-two `m` cycles
through a handful of values — `% 2` gives 0,1,0,1 forever. Use
[`monte_carlo::Rng::below`], which takes the high bits, for any small-integer
draw. It is fine for simulation and not cryptographically secure.

**Explicit time-stepping is conditionally stable.** Heat needs
`α Δt / Δx² ≤ 1/4` in 2-D and `1/6` in 3-D, so halving the grid spacing
*quarters* the time step. Waves and FDTD need the Courant condition, and the
limit is set by the fastest medium in the grid — for FDTD that means the
*smallest* relative permittivity, not vacuum.

**`pcg_jacobi`'s tolerance is relative to the norm of the right-hand side**,
not absolute. Passing `1e-13` when your data is at `1e8` asks for something
much weaker than you meant.

**Reference tables are room-temperature values.** Viscosity in particular can
change by a factor of several over a few tens of degrees; a single figure is a
starting point, not a datasheet.

**A specification reading is written down where one was needed.** Where a
definition is genuinely ambiguous — the Frobenius number with a unit coin,
Stern–Brocot indexing, which parenthesisation a unit string means — the choice
is stated in the doc comment rather than left implicit. If a result surprises
you, read the doc comment before assuming a bug.

---

[`acoustics`]: ../src/acoustics.rs
[`astrophysics`]: ../src/astrophysics/
[`astrophysics::kepler`]: ../src/astrophysics/kepler.rs
[`astrophysics::lambert`]: ../src/astrophysics/lambert.rs
[`astrophysics::maneuvers`]: ../src/astrophysics/maneuvers.rs
[`astrophysics::nbody`]: ../src/astrophysics/nbody.rs
[`audio`]: ../src/audio/
[`cfd`]: ../src/cfd/
[`classical`]: ../src/classical.rs
[`continuum_mechanics`]: ../src/continuum_mechanics.rs
[`core`]: ../src/core/
[`dsp`]: ../src/dsp/
[`dsp::iir`]: ../src/dsp/iir.rs
[`electromagnetism`]: ../src/electromagnetism.rs
[`electronics`]: ../src/electronics.rs
[`exact`]: ../src/exact/
[`fem`]: ../src/fem/
[`fem::fdtd`]: ../src/fem/fdtd.rs
[`fem::fem2d`]: ../src/fem/fem2d.rs
[`fem::spectral_pde`]: ../src/fem/spectral_pde.rs
[`finance`]: ../src/finance/
[`fluids`]: ../src/fluids.rs
[`learn`]: ../src/learn/
[`manifold`]: ../src/manifold/
[`mesh`]: ../src/mesh/
[`monte_carlo::Rng::below`]: ../src/monte_carlo/mod.rs
[`optimization`]: ../src/optimization/
[`propulsion`]: ../src/propulsion.rs
[`quantum`]: ../src/quantum/
[`resonance`]: ../src/resonance/
[`rf`]: ../src/rf.rs
[`signal_processing`]: ../src/signal_processing/
[`sim`]: ../src/sim/
[`sim::heat_sim`]: ../src/sim/heat_sim.rs
[`solid_mechanics`]: ../src/solid_mechanics.rs
[`spatial`]: ../src/spatial/
[`statistics`]: ../src/statistics/
[`thermodynamics`]: ../src/thermodynamics.rs
[`transforms`]: ../src/transforms/
[`transforms::fft`]: ../src/transforms/fft.rs
[`transforms::spectral`]: ../src/transforms/spectral.rs
[`transforms::wavelet`]: ../src/transforms/wavelet.rs
[`units`]: ../src/units/
