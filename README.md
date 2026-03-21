# rust_physics_engine

[![CI](https://github.com/Magic-Man-us/RustPhysicsEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/Magic-Man-us/RustPhysicsEngine/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Magic-Man-us/COVERAGE_GIST_ID/raw/coverage.json)](https://github.com/Magic-Man-us/RustPhysicsEngine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

A comprehensive, zero-dependency Rust library covering physics, mathematics, and engineering computation. Built for correctness — every public function validates its inputs, and the entire codebase sits at 99.98% test coverage with 1,659 tests.

## What's in it

### Classical & Continuum Mechanics
- **`classical`** — Newtonian mechanics: projectile motion, collisions, SHM, damping, resonance
- **`solid_mechanics`** — Stress, strain, elastic moduli, beam deflection, Mohr's circle
- **`continuum_mechanics`** — 3D Hooke's law, compliance matrices, plane stress/strain
- **`fluid_instabilities`** — Rayleigh-Taylor, Kelvin-Helmholtz, Jeans instability, Plateau-Rayleigh

### Thermodynamics & Statistical Mechanics
- **`thermodynamics`** — Ideal gas, Carnot, entropy, heat conduction, radiation, Nusselt/Biot/Grashof
- **`statistical_mechanics`** — Maxwell-Boltzmann, Boltzmann/Einstein/Debye models, diffusion, partition functions

### Electromagnetism & Electronics
- **`electromagnetism`** — Coulomb, Lorentz, Faraday, Maxwell, RLC circuits, transformers
- **`electronics`** — Semiconductors, diodes, MOSFETs, solar cells, PN junctions
- **`rf`** — Friis, skin depth, antenna gain, impedance, VSWR, Smith chart quantities
- **`photonics`** — Gaussian beams, fiber optics, ray transfer matrices, coherence, Fabry-Perot

### Waves, Optics & Acoustics
- **`waves`** — Doppler, standing waves, diffraction, Snell, seismic waves, dispersion
- **`optics`** — Lenses, mirrors, thin films, diffraction gratings, Rayleigh resolution
- **`acoustics`** — Sabine/Eyring reverberation, psychoacoustic scales (mel, bark, ERB), room modes, STC
- **`signal_processing`** — Waveform generation, FIR/IIR filters, convolution, windowing, resampling

### Relativity & Quantum
- **`relativity`** — Lorentz transformations, relativistic energy-momentum, time dilation, Doppler
- **`general_relativity`** — Schwarzschild metric, geodesics, frame dragging, cosmological distances
- **`quantum`** — De Broglie, uncertainty principle, particle-in-a-box, tunneling, Planck radiation
- **`particle_physics`** — Invariant mass, Rutherford scattering, Breit-Wigner, rapidity, Lorentz boosts

### Nuclear & Radiation
- **`nuclear`** — Decay chains, binding energy, Q-values, dosimetry
- **`neutronics`** — Criticality, diffusion, moderation, burnup, shielding
- **`radiation`** — Blackbody, Wien, Planck, radiative transfer, view factors
- **`plasma`** — Debye length, cyclotron/plasma frequencies, Alfven speed, beta, Larmor radius

### Astrophysics
- **`astrophysics::nbody`** — N-body gravitational simulation (leapfrog integrator)
- **`astrophysics::octree`** — Barnes-Hut tree for O(N log N) force computation
- **`astrophysics::orbital_elements`** — Keplerian elements from state vectors, orbit propagation
- **`astrophysics::gravitational_waves`** — Strain, luminosity, frequency, chirp mass
- **`astrophysics::tidal`** — Tidal forces, Roche limits, tidal tensors
- **`astrophysics::lagrange`** — L1–L5 Lagrange point computation
- **`astrophysics::habitable_zone`** — Habitable zone boundaries, tidal locking
- **`astrophysics::magnetosphere`** — Dipole fields, magnetopause radius, field line tracing
- **`astrophysics::collisions`** — Impact cratering, orbital debris, collision probabilities

### Fluids & Propulsion
- **`fluids`** — Bernoulli, Poiseuille, Reynolds, drag, capillarity, compressible flow
- **`propulsion`** — Tsiolkovsky rocket equation, Hohmann transfers, nozzle design, staging
- **`magnetohydrodynamics`** — Alfven waves, Hartmann flow, magnetic reconnection, pinch equilibria

### Chemistry & Biophysics
- **`chemistry`** — Arrhenius, Nernst, pH, electrochemistry, reaction kinetics
- **`biophysics`** — Nernst/Goldman potentials, Michaelis-Menten, Hill equation, hemodynamics

### Mathematics & Numerical Methods
- **`math`** — `Vec3` type, physical constants (NIST CODATA), linear algebra primitives
- **`linalg`** — 3x3 matrices, rotations, eigenvalue decomposition, SVD-like operations
- **`quaternion`** — Quaternion algebra, slerp/nlerp, axis-angle, Euler angle conversions
- **`numerical`** — Root finding (bisection, Newton, secant), integration (Simpson, Gauss-Legendre), cubic splines
- **`optimization`** — Golden section, Brent, Nelder-Mead, simulated annealing, linear regression, polynomial fitting
- **`statistics`** — Mean, variance, median, distributions (Gaussian, Poisson, exponential), DFT, power spectrum
- **`monte_carlo`** — MC integration, Metropolis-Hastings, Ising model, Langevin dynamics, random walks
- **`vector_calculus`** — Gradient, divergence, curl, Laplacian, Poisson solver, line/surface/volume integrals
- **`nonlinear`** — Logistic map, Lorenz/Rossler attractors, Lyapunov exponents, bifurcation diagrams
- **`information_theory`** — Shannon entropy, mutual information, KL divergence, channel capacity

### Geometry, Curves & Fractals
- **`geometry`** — Areas, volumes, perimeters for standard shapes, regular polygons
- **`curves`** — Conic sections, Bezier curves, arc length, curvature
- **`fractals`** — Mandelbrot, Julia, burning ship, Newton fractals, Barnsley fern, box counting
- **`trigonometry`** — Trig identities, hyperbolic functions, angle conversions, haversine

### Simulation Engines
- **`sim::rigid_body`** — 3D rigid body dynamics with quaternion orientation, Euler equations, collision response
- **`sim::fluid_sim`** — Column fluid, 1D shallow water, 2D incompressible Euler (pressure projection)
- **`sim::heat_sim`** — 2D/3D heat conduction (explicit finite difference), convection-diffusion
- **`sim::wave_sim`** — 1D/2D wave equation solvers, absorbing boundary conditions (Mur ABC)
- **`sim::em_sim`** — 1D/2D FDTD electromagnetic simulation, PEC/Mur boundaries, dielectric media
- **`sim::cloth_sim`** — Verlet integration cloth/rope, spring-damper constraints, collision

### Reference Data
- **`materials::elements`** — All 118 elements with atomic mass, density, melting/boiling points, thermal/electrical conductivity
- **`materials::common`** — Engineering materials (steel, aluminum, copper, etc.)
- **`materials::fluids`** — 16 common fluids with density, viscosity, surface tension, speed of sound
- **`materials::gases`** — Common gases with molar mass, specific heat ratio, thermal conductivity

### Utilities
- **`units`** — SI unit conversions (temperature, pressure, energy, length, speed, angle, etc.)
- **`color_science`** — RGB/HSV/HSL/XYZ, wavelength-to-color, blackbody color, CIE color difference
- **`control_systems`** — Transfer functions, step/impulse response, PID tuning, stability margins
- **`atmosphere`** — ISA model, barometric formula, lapse rates, humidity, wind shear

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rust_physics_engine = { path = "." }
```

```rust
use rust_physics_engine::classical::{projectile_range, kinetic_energy};
use rust_physics_engine::math::constants::{G, C, K_B};
use rust_physics_engine::thermodynamics::ideal_gas_pressure;
use rust_physics_engine::sim::rigid_body::RigidBody;

// Projectile on Earth
let range = projectile_range(50.0, 0.7854, 9.81); // v₀=50 m/s, θ=45°, g=9.81

// Ideal gas
let p = ideal_gas_pressure(2.0, 8.314, 300.0, 0.05); // n, R, T, V

// Rigid body simulation
let mut body = RigidBody::new_sphere(10.0, 0.5); // mass=10 kg, radius=0.5 m
body.apply_force(rust_physics_engine::math::Vec3::new(0.0, -98.1, 0.0));
body.step(0.01);
```

## Design

- **Zero dependencies** — pure Rust, no external crates
- **`f64` throughout** — double precision for all computation
- **Input validation** — every function asserts valid inputs (positive mass, non-zero denominators, physical bounds)
- **NIST constants** — physical constants from CODATA 2018/2019
- **99.98% line coverage** — 1,659 tests across 67 source files

## Building & Testing

```bash
cargo build              # Build
cargo test               # Run all 1,659 tests
cargo llvm-cov           # Coverage report (requires cargo-llvm-cov)
```

## License

MIT
