//! Electrons and phonons in crystals: bands, densities of states, transport,
//! and the standard model systems.
//!
//! Bloch's theorem is the organising fact. A potential with a lattice
//! translation symmetry has eigenstates labelled by a crystal momentum, so
//! the infinite problem reduces to one over a single Brillouin zone -- and
//! the spectrum breaks into bands separated by gaps. That the gaps exist at
//! all is the reason there are insulators; that they are absent at the Fermi
//! level is the reason there are metals; and everything about semiconductors
//! is the behaviour of a gap small enough for temperature to matter.
//!
//! Functions take `hbar` and the masses explicitly where a natural-unit
//! calculation is the point, and use SI constants where a number in
//! electronvolts or siemens is wanted.

use crate::error::GeomError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// The reduced Planck constant, in joule seconds.
const HBAR: f64 = 1.054_571_817e-34;
/// Boltzmann's constant, in joules per kelvin.
const BOLTZMANN: f64 = 1.380_649e-23;
/// The elementary charge, in coulombs.
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// The electron mass, in kilograms.
const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;
/// The vacuum permittivity, in farads per metre.
const EPSILON_0: f64 = 8.854_187_812_8e-12;

// ---------------------------------------------------------------------------
// Tight binding
// ---------------------------------------------------------------------------

/// A one-dimensional tight-binding chain, returning the energies ascending
/// and the matching eigenvectors as rows.
///
/// `on_site` gives each site's energy and `t_hop` the nearest-neighbour
/// amplitude. The whole band structure of a simple metal is this model with
/// the on-site energies equal.
///
/// # Errors
/// Returns an error for fewer than two sites, more than five hundred, or an
/// eigensolver failure.
pub fn tight_binding_1d(
    t_hop: f64,
    on_site: &[f64],
    periodic: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), GeomError> {
    let n = on_site.len();
    if !(2..=500).contains(&n) {
        return Err(GeomError::InvalidArgument("the chain needs 2 to 500 sites"));
    }
    if periodic {
        // A ring is no longer tridiagonal, so it goes through the dense
        // solver; a chain stays tridiagonal and does not.
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, on_site[i]);
            let next = (i + 1) % n;
            m.set(i, next, m.get(i, next) - t_hop);
            m.set(next, i, m.get(next, i) - t_hop);
        }
        let decomposition = crate::linalg::eigen::eigen_symmetric(&m, 1e-13, 300)
            .map_err(|_| GeomError::Degenerate("the tight-binding eigenproblem failed"))?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            decomposition.values[a]
                .partial_cmp(&decomposition.values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let values: Vec<f64> = order.iter().map(|&i| decomposition.values[i]).collect();
        let vectors: Vec<Vec<f64>> = order
            .iter()
            .map(|&i| (0..n).map(|k| decomposition.vectors.get(k, i)).collect())
            .collect();
        return Ok((values, vectors));
    }
    let off = vec![-t_hop; n - 1];
    crate::linalg::tridiagonal::eigen_symmetric_tridiagonal(on_site, &off)
        .map_err(|_| GeomError::Degenerate("the tight-binding eigenproblem failed"))
}

/// The tight-binding band of an infinite chain: `-2 t cos(k a)`.
///
/// The bandwidth is `4 t` whatever the lattice constant, and the effective
/// mass at the band bottom is `hbar^2 / (2 t a^2)` -- so a narrow band means
/// a heavy electron, which is the whole of why transition metal oxides
/// behave as they do.
#[must_use]
pub fn tight_binding_band_1d(k: f64, t_hop: f64, a: f64) -> f64 {
    -2.0 * t_hop * (k * a).cos()
}

/// The Su-Schrieffer-Heeger model: a dimerised chain with alternating
/// hoppings.
///
/// Returns the energies ascending and the eigenvectors as rows. The chain has
/// `2 n` sites, `n` unit cells of two.
///
/// # Errors
/// Returns an error for a bad cell count or an eigensolver failure.
pub fn ssh_model(cells: usize, t1: f64, t2: f64) -> Result<(Vec<f64>, Vec<Vec<f64>>), GeomError> {
    if !(2..=200).contains(&cells) {
        return Err(GeomError::InvalidArgument("the SSH chain needs 2 to 200 cells"));
    }
    let n = 2 * cells;
    let diag = vec![0.0; n];
    // Alternating intracell and intercell hoppings.
    let off: Vec<f64> = (0..n - 1)
        .map(|i| if i % 2 == 0 { -t1 } else { -t2 })
        .collect();
    crate::linalg::tridiagonal::eigen_symmetric_tridiagonal(&diag, &off)
        .map_err(|_| GeomError::Degenerate("the SSH eigenproblem failed"))
}

/// The SSH winding number: one in the topological phase, zero otherwise.
///
/// The invariant is a property of the *bulk* -- it is computed from the
/// Hamiltonian's winding in momentum space with no reference to any edge --
/// and yet it predicts the number of protected edge states. That is the
/// bulk-boundary correspondence, and it is why topological states survive
/// disorder that would destroy an ordinary bound state.
#[must_use]
pub fn ssh_winding_number(t1: f64, t2: f64) -> i32 {
    i32::from(t2.abs() > t1.abs())
}

/// The number of near-zero-energy edge states of a finite SSH chain.
///
/// # Errors
/// Returns an error for a bad cell count.
pub fn ssh_edge_states(cells: usize, t1: f64, t2: f64) -> Result<usize, GeomError> {
    let (energies, _) = ssh_model(cells, t1, t2)?;
    // The bulk gap is 2 |t1 - t2|; anything well inside it is an edge state.
    let gap = 2.0 * (t1.abs() - t2.abs()).abs();
    let threshold = (gap / 4.0).max(1e-9);
    Ok(energies.iter().filter(|e| e.abs() < threshold).count())
}

/// The spectrum of a tight-binding square lattice with open boundaries.
///
/// The eigenvalues are separable: `-2t(cos(k_x a) + cos(k_y a))` with the
/// allowed momenta set by the box, so no diagonalisation is needed. That
/// separability is exactly why the square lattice is the standard sanity
/// check for a lattice code.
///
/// # Errors
/// Returns an error for a bad lattice size.
pub fn tight_binding_square(nx: usize, ny: usize, t_hop: f64) -> Result<Vec<f64>, GeomError> {
    if nx == 0 || ny == 0 || nx * ny > 40_000 {
        return Err(GeomError::InvalidArgument("the lattice size is out of range"));
    }
    let mut out = Vec::with_capacity(nx * ny);
    for i in 1..=nx {
        for j in 1..=ny {
            let kx = i as f64 * std::f64::consts::PI / (nx + 1) as f64;
            let ky = j as f64 * std::f64::consts::PI / (ny + 1) as f64;
            out.push(-2.0 * t_hop * (kx.cos() + ky.cos()));
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

/// The two graphene bands at a point of the Brillouin zone, in units where
/// the lattice constant is one.
///
/// The bands touch at the corners of the zone, and near them the dispersion
/// is *linear* rather than quadratic -- the electrons behave as massless
/// Dirac particles. Nothing about that requires relativity; it is a
/// consequence of the honeycomb's two-atom basis and its symmetry.
#[must_use]
pub fn graphene_dispersion(kx: f64, ky: f64, t_hop: f64) -> (f64, f64) {
    // The three nearest-neighbour vectors of the honeycomb lattice.
    let sqrt3 = 3.0f64.sqrt();
    let magnitude = (1.0
        + 4.0 * (sqrt3 * ky / 2.0).cos() * (3.0 * kx / 2.0).cos()
        + 4.0 * (sqrt3 * ky / 2.0).cos().powi(2))
    .max(0.0)
    .sqrt();
    (-t_hop * magnitude, t_hop * magnitude)
}

/// The six Dirac points of graphene, in the same units.
#[must_use]
pub fn dirac_points_graphene() -> Vec<(f64, f64)> {
    let sqrt3 = 3.0f64.sqrt();
    let a = 2.0 * std::f64::consts::PI / 3.0;
    let b = 2.0 * std::f64::consts::PI / (3.0 * sqrt3);
    vec![
        (a, b),
        (a, -b),
        (-a, b),
        (-a, -b),
        (0.0, 2.0 * b),
        (0.0, -2.0 * b),
    ]
}

// ---------------------------------------------------------------------------
// Kronig-Penney
// ---------------------------------------------------------------------------

/// The Kronig-Penney dispersion function: the right-hand side of
/// `cos(k L) = f(E)`.
///
/// Bands are where `|f| <= 1`, since only there does a real crystal momentum
/// exist. Where `|f| > 1` the momentum is complex and the states decay --
/// that is a gap, and it is the whole mechanism by which a periodic potential
/// forbids energies.
///
/// The well has width `a` and depth zero, the barrier width `b` and height
/// `v0`.
///
/// # Errors
/// Returns an error for non-positive widths, mass, or `hbar`.
pub fn kronig_penney(
    v0: f64,
    a: f64,
    b: f64,
    energy: f64,
    mass: f64,
    hbar: f64,
) -> Result<f64, GeomError> {
    if !(a > 0.0) || !(b > 0.0) || !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("kronig_penney: bad parameters"));
    }
    let factor = 2.0 * mass / (hbar * hbar);
    let alpha = (factor * energy).abs().sqrt();
    if energy < v0 {
        // Below the barrier: hyperbolic inside it.
        let beta = (factor * (v0 - energy)).sqrt();
        if alpha == 0.0 || beta == 0.0 {
            return Ok(f64::INFINITY);
        }
        let term = (beta * beta - alpha * alpha) / (2.0 * alpha * beta);
        Ok(term * (beta * b).sinh() * (alpha * a).sin() + (beta * b).cosh() * (alpha * a).cos())
    } else {
        let beta = (factor * (energy - v0)).sqrt();
        if alpha == 0.0 || beta == 0.0 {
            return Ok(f64::INFINITY);
        }
        let term = -(beta * beta + alpha * alpha) / (2.0 * alpha * beta);
        Ok(term * (beta * b).sin() * (alpha * a).sin() + (beta * b).cos() * (alpha * a).cos())
    }
}

/// The allowed energy bands of a Kronig-Penney lattice, as intervals.
///
/// # Errors
/// Returns an error for a bad range or sample count.
pub fn kronig_penney_bands(
    v0: f64,
    a: f64,
    b: f64,
    energy_range: (f64, f64),
    samples: usize,
    mass: f64,
    hbar: f64,
) -> Result<Vec<(f64, f64)>, GeomError> {
    let (lo, hi) = energy_range;
    if !(hi > lo) || samples < 2 {
        return Err(GeomError::InvalidArgument("kronig_penney_bands: bad range"));
    }
    let mut bands: Vec<(f64, f64)> = Vec::new();
    let mut inside: Option<f64> = None;
    for k in 0..=samples {
        let energy = lo + (hi - lo) * k as f64 / samples as f64;
        let value = kronig_penney(v0, a, b, energy, mass, hbar)?;
        let allowed = value.abs() <= 1.0;
        match (allowed, inside) {
            (true, None) => inside = Some(energy),
            (false, Some(start)) => {
                bands.push((start, energy));
                inside = None;
            }
            _ => {}
        }
    }
    if let Some(start) = inside {
        bands.push((start, hi));
    }
    Ok(bands)
}

// ---------------------------------------------------------------------------
// Densities of states and occupations
// ---------------------------------------------------------------------------

/// The free-electron density of states per unit volume in one dimension.
///
/// Spin degeneracy is included, as it is in the two- and three-dimensional
/// versions below: integrating any of them up to the Fermi energy gives the
/// electron density directly, with no further factor of two.
///
/// Diverges as `1 / sqrt(E)` at the band bottom -- a van Hove singularity,
/// and the reason one-dimensional systems are so unstable to any interaction
/// at all.
///
/// # Errors
/// Returns an error for a non-positive mass or `hbar`.
pub fn density_of_states_1d_free(energy: f64, mass: f64, hbar: f64) -> Result<f64, GeomError> {
    if !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("the mass and hbar must be positive"));
    }
    if energy <= 0.0 {
        return Ok(0.0);
    }
    Ok((2.0 * mass).sqrt() / (std::f64::consts::PI * hbar * energy.sqrt()))
}

/// The free-electron density of states in two dimensions: a constant.
///
/// Energy independent above the band bottom, which is what makes a
/// two-dimensional electron gas the clean setting for the quantum Hall
/// effect.
///
/// # Errors
/// Returns an error for a non-positive mass or `hbar`.
pub fn density_of_states_2d_free(energy: f64, mass: f64, hbar: f64) -> Result<f64, GeomError> {
    if !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("the mass and hbar must be positive"));
    }
    if energy <= 0.0 {
        return Ok(0.0);
    }
    Ok(mass / (std::f64::consts::PI * hbar * hbar))
}

/// The free-electron density of states in three dimensions, going as
/// `sqrt(E)`.
///
/// # Errors
/// Returns an error for a non-positive mass or `hbar`.
pub fn density_of_states_3d_free(energy: f64, mass: f64, hbar: f64) -> Result<f64, GeomError> {
    if !(mass > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("the mass and hbar must be positive"));
    }
    if energy <= 0.0 {
        return Ok(0.0);
    }
    let prefactor = (2.0 * mass).powf(1.5) / (2.0 * std::f64::consts::PI.powi(2) * hbar.powi(3));
    Ok(prefactor * energy.sqrt())
}

/// A density of states from a list of levels, broadened by a Gaussian.
///
/// # Errors
/// Returns an error for an empty list, a non-positive width, or too few
/// points.
pub fn dos_from_bands(
    levels: &[f64],
    sigma: f64,
    points: usize,
) -> Result<Vec<(f64, f64)>, GeomError> {
    if levels.is_empty() || !(sigma > 0.0) || points < 2 {
        return Err(GeomError::InvalidArgument("dos_from_bands: bad input"));
    }
    let lo = levels.iter().copied().fold(f64::INFINITY, f64::min) - 4.0 * sigma;
    let hi = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 4.0 * sigma;
    let norm = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    Ok((0..points)
        .map(|k| {
            let energy = lo + (hi - lo) * k as f64 / (points - 1) as f64;
            let density: f64 = levels
                .iter()
                .map(|e| norm * (-(energy - e).powi(2) / (2.0 * sigma * sigma)).exp())
                .sum();
            (energy, density)
        })
        .collect())
}

/// The Fermi-Dirac occupation.
///
/// # Errors
/// Returns an error for a negative temperature.
pub fn fermi_dirac(energy: f64, mu: f64, temperature: f64) -> Result<f64, GeomError> {
    if temperature < 0.0 {
        return Err(GeomError::InvalidArgument("the temperature cannot be negative"));
    }
    if temperature == 0.0 {
        return Ok(if energy < mu {
            1.0
        } else if energy > mu {
            0.0
        } else {
            0.5
        });
    }
    let x = (energy - mu) / (BOLTZMANN * temperature);
    // Written to avoid overflow at either extreme.
    Ok(if x > 0.0 {
        let e = (-x).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + x.exp())
    })
}

/// The Bose-Einstein occupation.
///
/// Diverges as the energy approaches the chemical potential, which is
/// condensation: the ground state's occupation is not bounded by one, and in
/// three dimensions it takes a macroscopic share below a finite temperature.
///
/// # Errors
/// Returns an error for a negative temperature or an energy at or below the
/// chemical potential.
pub fn bose_einstein(energy: f64, mu: f64, temperature: f64) -> Result<f64, GeomError> {
    if temperature < 0.0 {
        return Err(GeomError::InvalidArgument("the temperature cannot be negative"));
    }
    if energy <= mu {
        return Err(GeomError::InvalidArgument("bosons require the energy above mu"));
    }
    if temperature == 0.0 {
        return Ok(0.0);
    }
    let x = (energy - mu) / (BOLTZMANN * temperature);
    Ok(1.0 / (x.exp() - 1.0))
}

/// The Fermi energy of a free electron gas at the given number density.
///
/// # Errors
/// Returns an error for a non-positive density or mass.
pub fn fermi_energy_free(density: f64, mass: f64) -> Result<f64, GeomError> {
    if !(density > 0.0) || !(mass > 0.0) {
        return Err(GeomError::InvalidArgument("the density and mass must be positive"));
    }
    let k_f = (3.0 * std::f64::consts::PI * std::f64::consts::PI * density).powf(1.0 / 3.0);
    Ok(HBAR * HBAR * k_f * k_f / (2.0 * mass))
}

/// The Sommerfeld electronic heat capacity per electron.
///
/// Linear in temperature, and smaller than the classical `3k/2` by a factor
/// of order `T / T_F` -- which resolves the nineteenth-century puzzle of why
/// metals' electrons contribute almost nothing to the heat capacity despite
/// carrying the current. Only those within `kT` of the Fermi surface can
/// absorb energy at all.
///
/// # Errors
/// Returns an error for a non-positive Fermi temperature.
pub fn sommerfeld_heat_capacity(temperature: f64, fermi_temperature: f64) -> Result<f64, GeomError> {
    if !(fermi_temperature > 0.0) || temperature < 0.0 {
        return Err(GeomError::InvalidArgument("sommerfeld_heat_capacity: bad temperatures"));
    }
    Ok(std::f64::consts::PI * std::f64::consts::PI / 2.0 * BOLTZMANN * temperature
        / fermi_temperature)
}

/// The Debye heat capacity per atom.
///
/// Goes as `T^3` at low temperature and to the classical `3k` at high --
/// Dulong and Petit's law. The cube is the count of phonon modes thermally
/// accessible, and it is one of the earliest quantitative successes of
/// quantum theory applied to solids.
///
/// # Errors
/// Returns an error for a non-positive Debye temperature.
pub fn debye_heat_capacity(temperature: f64, debye_temperature: f64) -> Result<f64, GeomError> {
    if !(debye_temperature > 0.0) || temperature < 0.0 {
        return Err(GeomError::InvalidArgument("debye_heat_capacity: bad temperatures"));
    }
    if temperature == 0.0 {
        return Ok(0.0);
    }
    let ratio = temperature / debye_temperature;
    let upper = 1.0 / ratio;
    // The Debye integral, by midpoint quadrature; the integrand is smooth on
    // the whole range once the removable singularity at zero is handled.
    let samples = 4000usize;
    let h = upper / samples as f64;
    let integral: f64 = (0..samples)
        .map(|k| {
            let x = (k as f64 + 0.5) * h;
            let e = x.exp();
            if !e.is_finite() {
                return 0.0;
            }
            x.powi(4) * e / (e - 1.0).powi(2)
        })
        .sum::<f64>()
        * h;
    Ok(9.0 * BOLTZMANN * ratio.powi(3) * integral)
}

/// The Einstein heat capacity per atom, from a single vibrational frequency.
///
/// Falls exponentially at low temperature rather than as `T^3`, which is
/// exactly where the model fails and Debye's succeeds: a single frequency
/// leaves no low-energy modes to excite, and a real solid has acoustic
/// phonons of arbitrarily low frequency.
///
/// # Errors
/// Returns an error for a non-positive Einstein temperature.
pub fn einstein_heat_capacity(
    temperature: f64,
    einstein_temperature: f64,
) -> Result<f64, GeomError> {
    if !(einstein_temperature > 0.0) || temperature < 0.0 {
        return Err(GeomError::InvalidArgument("einstein_heat_capacity: bad temperatures"));
    }
    if temperature == 0.0 {
        return Ok(0.0);
    }
    let x = einstein_temperature / temperature;
    if x > 700.0 {
        return Ok(0.0);
    }
    let e = x.exp();
    Ok(3.0 * BOLTZMANN * x * x * e / (e - 1.0).powi(2))
}

// ---------------------------------------------------------------------------
// Phonons and fields
// ---------------------------------------------------------------------------

/// The phonon dispersion of a monatomic chain.
///
/// Linear at long wavelength -- sound -- and flattening at the zone boundary,
/// where the group velocity vanishes and the mode becomes a standing wave.
///
/// # Panics
/// Panics unless the spring constant and mass are positive.
#[must_use]
pub fn phonon_dispersion_1d_monatomic(k: f64, spring: f64, mass: f64, a: f64) -> f64 {
    assert!(spring > 0.0 && mass > 0.0, "the spring constant and mass must be positive");
    2.0 * (spring / mass).sqrt() * (k * a / 2.0).sin().abs()
}

/// The two phonon branches of a diatomic chain, acoustic first.
///
/// The gap between them at the zone boundary is the mass difference made
/// audible: a diatomic crystal has optical modes that a monatomic one does
/// not, and they are what infrared spectroscopy sees.
///
/// # Panics
/// Panics unless the spring constant and both masses are positive.
#[must_use]
pub fn phonon_dispersion_1d_diatomic(
    k: f64,
    spring: f64,
    m1: f64,
    m2: f64,
    a: f64,
) -> (f64, f64) {
    assert!(spring > 0.0 && m1 > 0.0 && m2 > 0.0, "the parameters must be positive");
    let sum = 1.0 / m1 + 1.0 / m2;
    let inner = sum * sum - 4.0 * (k * a).sin().powi(2) / (m1 * m2);
    let root = inner.max(0.0).sqrt();
    let acoustic = (spring * (sum - root)).max(0.0).sqrt();
    let optical = (spring * (sum + root)).max(0.0).sqrt();
    (acoustic, optical)
}

/// The Bloch oscillation period of an electron in a static field.
///
/// An electron in a perfect crystal under a constant force does not
/// accelerate away: it traverses the Brillouin zone and comes back, so it
/// *oscillates*. Ordinary conductors never show this because scattering
/// intervenes long before a period completes; superlattices, with their much
/// smaller zones, do.
///
/// # Errors
/// Returns an error for a non-positive field or lattice constant.
pub fn bloch_oscillation_period(field: f64, a: f64) -> Result<f64, GeomError> {
    if !(field > 0.0) || !(a > 0.0) {
        return Err(GeomError::InvalidArgument("the field and spacing must be positive"));
    }
    Ok(2.0 * std::f64::consts::PI * HBAR / (ELEMENTARY_CHARGE * field * a))
}

/// The energy of the `n`-th Landau level.
///
/// Equally spaced by `hbar omega_c`, with a zero-point half. The spacing
/// depends on the field and not on the level, which is what makes the
/// magneto-oscillations periodic in `1 / B` and lets a Fermi surface be
/// measured.
///
/// # Errors
/// Returns an error for a non-positive field or mass.
pub fn landau_levels(field: f64, n: usize, mass: f64) -> Result<f64, GeomError> {
    if !(field > 0.0) || !(mass > 0.0) {
        return Err(GeomError::InvalidArgument("the field and mass must be positive"));
    }
    let cyclotron = ELEMENTARY_CHARGE * field / mass;
    Ok((n as f64 + 0.5) * HBAR * cyclotron)
}

/// The Hofstadter spectrum: the energies of a square lattice at each rational
/// flux `p / q`, as `(flux, energy)` pairs.
///
/// The famous butterfly. At flux `p / q` the magnetic unit cell holds `q`
/// sites, so the band splits into `q` sub-bands -- and because that count
/// depends on the *denominator*, the spectrum is discontinuous in the flux at
/// every rational. It is the first place a fractal appeared in a physical
/// spectrum.
///
/// # Errors
/// Returns an error for a bad denominator bound or momentum sample count.
pub fn hofstadter_butterfly(q_max: usize, k_samples: usize) -> Result<Vec<(f64, f64)>, GeomError> {
    if !(2..=40).contains(&q_max) || k_samples == 0 {
        return Err(GeomError::InvalidArgument("hofstadter_butterfly: bad parameters"));
    }
    let mut out = Vec::new();
    for q in 2..=q_max {
        for p in 1..q {
            if gcd(p, q) != 1 {
                continue;
            }
            let flux = p as f64 / q as f64;
            // Harper's equation: a q x q tridiagonal matrix with a phase in
            // the corners, sampled over the magnetic Brillouin zone.
            for s in 0..k_samples {
                let ky = 2.0 * std::f64::consts::PI * s as f64 / (k_samples * q) as f64;
                let mut m = Matrix::zeros(q, q);
                for j in 0..q {
                    m.set(
                        j,
                        j,
                        2.0 * (2.0 * std::f64::consts::PI * flux * j as f64 + ky).cos(),
                    );
                    let next = (j + 1) % q;
                    if q > 2 {
                        m.set(j, next, m.get(j, next) + 1.0);
                        m.set(next, j, m.get(next, j) + 1.0);
                    } else if j == 0 {
                        m.set(0, 1, 2.0);
                        m.set(1, 0, 2.0);
                    }
                }
                let decomposition = crate::linalg::eigen::eigen_symmetric(&m, 1e-12, 200)
                    .map_err(|_| GeomError::Degenerate("the Harper eigenproblem failed"))?;
                for e in &decomposition.values {
                    out.push((flux, *e));
                }
            }
        }
    }
    Ok(out)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// The Hall conductance of `n` filled Landau levels, in siemens.
///
/// Quantised in units of `e^2 / h` to a part in a billion, in samples whose
/// disorder is uncontrolled and whose geometry is irregular. That the answer
/// depends on nothing but fundamental constants is why it defines the ohm.
#[must_use]
pub fn quantum_hall_conductance(filled: usize) -> f64 {
    filled as f64 * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
        / (2.0 * std::f64::consts::PI * HBAR)
}

/// The Drude conductivity.
///
/// # Errors
/// Returns an error for a non-positive relaxation time or mass.
pub fn drude_conductivity(density: f64, tau: f64, mass: f64) -> Result<f64, GeomError> {
    if !(tau > 0.0) || !(mass > 0.0) || density < 0.0 {
        return Err(GeomError::InvalidArgument("drude_conductivity: bad parameters"));
    }
    Ok(density * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * tau / mass)
}

/// The Hall coefficient of a single-carrier conductor.
///
/// Its *sign* is the useful part: positive for holes and negative for
/// electrons, so a Hall measurement says which carries the current -- a fact
/// no conductivity measurement can supply.
///
/// # Errors
/// Returns an error for zero density.
pub fn hall_coefficient(density: f64, charge: f64) -> Result<f64, GeomError> {
    if density == 0.0 || charge == 0.0 {
        return Err(GeomError::InvalidArgument("hall_coefficient needs carriers"));
    }
    Ok(1.0 / (density * charge))
}

/// The effective mass at a point of a band, from its curvature.
///
/// `m* = hbar^2 / (d^2 E / dk^2)`, which can be negative near a band top --
/// and a negative effective mass is precisely what a hole is.
///
/// # Errors
/// Returns an error for a non-positive step or a flat band.
pub fn effective_mass_from_band(
    band: &dyn Fn(f64) -> f64,
    k0: f64,
    h: f64,
) -> Result<f64, GeomError> {
    if !(h > 0.0) {
        return Err(GeomError::InvalidArgument("the step must be positive"));
    }
    let curvature = (band(k0 + h) - 2.0 * band(k0) + band(k0 - h)) / (h * h);
    // The flatness test has to be relative to the band's own scale. An
    // absolute threshold silently reports every band in SI units as flat,
    // since a real band's curvature there is around 1e-38.
    let scale = (band(k0 + h).abs() + band(k0).abs() + band(k0 - h).abs()) / (h * h);
    if curvature.abs() <= 1e-12 * scale {
        return Err(GeomError::Degenerate("the band is flat here"));
    }
    Ok(HBAR * HBAR / curvature)
}

// ---------------------------------------------------------------------------
// Semiconductors and superconductors
// ---------------------------------------------------------------------------

/// The intrinsic carrier density of a semiconductor, per cubic metre.
///
/// The exponential in half the gap is what makes semiconductor conductivity
/// so temperature sensitive: silicon's carrier density roughly doubles every
/// eight kelvin at room temperature.
///
/// # Errors
/// Returns an error for a non-positive temperature or mass.
pub fn semiconductor_carrier_density(
    gap_ev: f64,
    temperature: f64,
    m_electron: f64,
    m_hole: f64,
) -> Result<f64, GeomError> {
    if !(temperature > 0.0) || !(m_electron > 0.0) || !(m_hole > 0.0) {
        return Err(GeomError::InvalidArgument("semiconductor_carrier_density: bad parameters"));
    }
    let kt = BOLTZMANN * temperature;
    let prefactor = |m: f64| 2.0 * (m * kt / (2.0 * std::f64::consts::PI * HBAR * HBAR)).powf(1.5);
    let nc = prefactor(m_electron * ELECTRON_MASS);
    let nv = prefactor(m_hole * ELECTRON_MASS);
    Ok((nc * nv).sqrt() * (-gap_ev * ELEMENTARY_CHARGE / (2.0 * kt)).exp())
}

/// The built-in potential of a p-n junction, in volts.
///
/// # Errors
/// Returns an error for non-positive doping, intrinsic density, or
/// temperature.
pub fn pn_junction_builtin(
    acceptors: f64,
    donors: f64,
    intrinsic: f64,
    temperature: f64,
) -> Result<f64, GeomError> {
    if !(acceptors > 0.0) || !(donors > 0.0) || !(intrinsic > 0.0) || !(temperature > 0.0) {
        return Err(GeomError::InvalidArgument("pn_junction_builtin: bad parameters"));
    }
    let thermal = BOLTZMANN * temperature / ELEMENTARY_CHARGE;
    Ok(thermal * (acceptors * donors / (intrinsic * intrinsic)).ln())
}

/// The depletion width of an abrupt p-n junction, in metres.
///
/// # Errors
/// Returns an error for non-positive doping or permittivity.
pub fn depletion_width(
    built_in: f64,
    acceptors: f64,
    donors: f64,
    relative_permittivity: f64,
) -> Result<f64, GeomError> {
    if !(acceptors > 0.0) || !(donors > 0.0) || !(relative_permittivity > 0.0) || built_in < 0.0 {
        return Err(GeomError::InvalidArgument("depletion_width: bad parameters"));
    }
    let epsilon = relative_permittivity * EPSILON_0;
    Ok((2.0 * epsilon * built_in / ELEMENTARY_CHARGE
        * (1.0 / acceptors + 1.0 / donors))
        .sqrt())
}

/// The BCS energy gap at temperature `t`, relative to its value at zero.
///
/// Solved from the gap equation, which is self-consistent: the gap appears on
/// both sides, so it has the trivial solution zero above the critical
/// temperature and a non-zero one below. That the transition is continuous
/// and the gap opens as `sqrt(1 - T / Tc)` is a prediction of the theory, not
/// an input to it.
///
/// # Errors
/// Returns an error for a non-positive critical temperature.
pub fn bcs_gap_equation(temperature: f64, critical_temperature: f64) -> Result<f64, GeomError> {
    if !(critical_temperature > 0.0) || temperature < 0.0 {
        return Err(GeomError::InvalidArgument("bcs_gap_equation: bad temperatures"));
    }
    if temperature >= critical_temperature {
        return Ok(0.0);
    }
    let t = temperature / critical_temperature;
    if t <= 0.0 {
        return Ok(1.0);
    }
    // The standard interpolation of the numerical solution,
    // `tanh(1.74 sqrt(Tc / T - 1))`, which is exact in both limits: it tends
    // to one at zero temperature and to `1.74 sqrt(1 - T / Tc)` at the
    // transition, reproducing the square-root opening the theory predicts.
    Ok((1.74 * (1.0 / t - 1.0).max(0.0).sqrt()).tanh())
}

/// The BCS critical temperature from the coupling and the Debye frequency.
///
/// `1.14 theta_D exp(-1 / lambda)`. The exponential in the reciprocal
/// coupling has no expansion about zero coupling, which is why
/// superconductivity could not be found by perturbation theory and took forty
/// years to explain.
///
/// # Errors
/// Returns an error for a non-positive coupling or Debye temperature.
pub fn bcs_tc_from_coupling(coupling: f64, debye_temperature: f64) -> Result<f64, GeomError> {
    if !(coupling > 0.0) || !(debye_temperature > 0.0) {
        return Err(GeomError::InvalidArgument("bcs_tc_from_coupling: bad parameters"));
    }
    Ok(1.14 * debye_temperature * (-1.0 / coupling).exp())
}

/// The DC Josephson current across a junction.
///
/// A supercurrent flows with no voltage at all, set only by the phase
/// difference across the barrier. It is the most direct evidence that the
/// superconducting order parameter has a phase and that the phase is
/// physical.
#[must_use]
pub fn josephson_current(critical_current: f64, phase: f64) -> f64 {
    critical_current * phase.sin()
}

/// The AC Josephson frequency at a given voltage: `2 e V / h`.
///
/// About 484 terahertz per volt, and known to a part in `10^10` -- which is
/// why the Josephson effect defines the volt.
#[must_use]
pub fn josephson_frequency(voltage: f64) -> f64 {
    2.0 * ELEMENTARY_CHARGE * voltage / (2.0 * std::f64::consts::PI * HBAR)
}

/// The localisation length of a disordered one-dimensional chain, in lattice
/// sites.
///
/// Every state in one dimension is localised for any disorder whatever, which
/// is the sharpest statement in the subject: there is no mobility edge and no
/// metallic phase, however weak the randomness. The length is extracted as
/// the reciprocal Lyapunov exponent of the transfer matrix product.
///
/// # Errors
/// Returns an error for a bad chain length, disorder, or trial count.
pub fn anderson_localization_1d(
    n: usize,
    disorder: f64,
    energy: f64,
    trials: usize,
    rng: &mut Rng,
) -> Result<f64, GeomError> {
    if n < 10 || !(disorder > 0.0) || trials == 0 {
        return Err(GeomError::InvalidArgument("anderson_localization_1d: bad parameters"));
    }
    let mut total = 0.0;
    for _ in 0..trials {
        // The transfer matrix of the Anderson chain, with the log of the
        // vector's growth accumulated to avoid overflow.
        let (mut a, mut b) = (1.0f64, 0.0f64);
        let mut log_growth = 0.0;
        for _ in 0..n {
            let on_site = disorder * (rng.next_f64() - 0.5);
            let next = (energy - on_site) * a - b;
            b = a;
            a = next;
            let magnitude = a.hypot(b);
            if magnitude > 0.0 {
                log_growth += magnitude.ln();
                a /= magnitude;
                b /= magnitude;
            }
        }
        total += log_growth / n as f64;
    }
    let lyapunov = total / trials as f64;
    if lyapunov <= 0.0 {
        return Err(GeomError::Degenerate("the Lyapunov exponent did not come out positive"));
    }
    Ok(1.0 / lyapunov)
}

/// The Landauer conductance of a set of transmission channels, in siemens.
///
/// Conductance is transmission: a ballistic channel with perfect transmission
/// carries `2 e^2 / h` and no more, so even a perfect wire has a finite
/// resistance. That resistance is not dissipation in the wire -- it is the
/// cost of matching a few channels to the infinitely many in the leads.
///
/// # Errors
/// Returns an error if a transmission is outside `[0, 1]`.
pub fn conductance_landauer(transmissions: &[f64]) -> Result<f64, GeomError> {
    if transmissions.iter().any(|t| !(0.0..=1.0).contains(t)) {
        return Err(GeomError::InvalidArgument("a transmission is not a probability"));
    }
    let quantum = 2.0 * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
        / (2.0 * std::f64::consts::PI * HBAR);
    Ok(quantum * transmissions.iter().sum::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn relative(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    // -----------------------------------------------------------------
    // Tight binding
    // -----------------------------------------------------------------

    #[test]
    fn the_tight_binding_chain_matches_its_closed_form_spectrum() {
        // An open chain of n sites has energies -2t cos(m pi / (n + 1)),
        // exactly. A ring has -2t cos(2 pi m / n), also exactly. The two
        // differ, and mixing them up is the commonest boundary-condition
        // error there is.
        let t = 1.3f64;
        for n in [2usize, 5, 12, 40] {
            let (energies, vectors) = tight_binding_1d(t, &vec![0.0; n], false).unwrap();
            let mut expected: Vec<f64> = (1..=n)
                .map(|m| -2.0 * t * (m as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
                .collect();
            expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for (got, want) in energies.iter().zip(&expected) {
                assert!(close(*got, *want, 1e-9), "open n = {n}: {energies:?} against {expected:?}");
            }
            // The eigenvectors are orthonormal.
            for i in 0..n {
                for j in 0..n {
                    let overlap: f64 =
                        vectors[i].iter().zip(&vectors[j]).map(|(a, b)| a * b).sum();
                    assert!(close(overlap, f64::from(i == j), 1e-9));
                }
            }

            if n > 2 {
                let (ring, _) = tight_binding_1d(t, &vec![0.0; n], true).unwrap();
                let mut expected: Vec<f64> = (0..n)
                    .map(|m| -2.0 * t * (2.0 * std::f64::consts::PI * m as f64 / n as f64).cos())
                    .collect();
                expected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for (got, want) in ring.iter().zip(&expected) {
                    assert!(close(*got, *want, 1e-8), "ring n = {n}: {ring:?} against {expected:?}");
                }
            }
        }
        // The band and the finite chain agree: the chain's levels sample the
        // band at the allowed momenta.
        let n = 60usize;
        let (energies, _) = tight_binding_1d(t, &vec![0.0; n], false).unwrap();
        for m in 1..=n {
            let k = m as f64 * std::f64::consts::PI / ((n + 1) as f64);
            let band = tight_binding_band_1d(k, t, 1.0);
            assert!(
                energies.iter().any(|e| close(*e, band, 1e-9)),
                "the band value {band} at k = {k} is not a level"
            );
        }
        // The bandwidth is 4t whatever the lattice constant.
        let bottom = tight_binding_band_1d(0.0, t, 2.7);
        let top = tight_binding_band_1d(std::f64::consts::PI / 2.7, t, 2.7);
        assert!(close(top - bottom, 4.0 * t, 1e-9), "the bandwidth is {}", top - bottom);

        assert!(tight_binding_1d(t, &[1.0], false).is_err());
        assert!(tight_binding_1d(t, &vec![0.0; 600], false).is_err());
    }

    #[test]
    fn the_ssh_chain_has_edge_states_exactly_when_its_winding_number_says_so() {
        // Bulk-boundary correspondence, tested as a correspondence: the
        // invariant is computed from the couplings alone and the edge states
        // are counted from the spectrum, with nothing shared between them.
        for (t1, t2) in [(1.0f64, 2.0f64), (2.0, 1.0), (0.5, 3.0), (3.0, 0.5), (1.0, 1.0)] {
            let winding = ssh_winding_number(t1, t2);
            let states = ssh_edge_states(40, t1, t2).unwrap();
            if (t1 - t2).abs() < 1e-12 {
                // At the transition the gap closes and the question has no
                // answer, so nothing is asserted about the count.
                assert_eq!(winding, 0);
                continue;
            }
            assert_eq!(
                states,
                2 * winding as usize,
                "t1 = {t1}, t2 = {t2}: winding {winding} but {states} edge states"
            );
        }

        // The edge states are exponentially localised at the ends.
        let (energies, vectors) = ssh_model(40, 0.5, 2.0).unwrap();
        let zero_modes: Vec<usize> = (0..energies.len())
            .filter(|&i| energies[i].abs() < 0.1)
            .collect();
        assert_eq!(zero_modes.len(), 2, "expected two zero modes, got {}", zero_modes.len());
        for &i in &zero_modes {
            let weight: f64 = vectors[i].iter().map(|c| c * c).sum();
            let edges: f64 = vectors[i][..6].iter().map(|c| c * c).sum::<f64>()
                + vectors[i][74..].iter().map(|c| c * c).sum::<f64>();
            assert!(
                edges / weight > 0.9,
                "the zero mode has only {} of its weight at the edges",
                edges / weight
            );
        }
        // And the gap is 2 |t1 - t2| as advertised.
        let gap = energies
            .iter()
            .filter(|e| **e > 0.2)
            .fold(f64::INFINITY, |acc, e| acc.min(*e))
            * 2.0;
        assert!(close(gap, 2.0 * 1.5, 0.05), "the gap is {gap}");

        assert!(ssh_model(1, 1.0, 2.0).is_err());
        assert!(ssh_edge_states(1, 1.0, 2.0).is_err());
    }

    #[test]
    fn the_square_lattice_is_separable_and_graphene_closes_its_gap_at_the_dirac_points() {
        // The square lattice's spectrum runs from -4t to 4t, and its levels
        // are sums of two one-dimensional ones.
        let t = 1.0f64;
        let levels = tight_binding_square(6, 6, t).unwrap();
        assert_eq!(levels.len(), 36);
        assert!(levels.windows(2).all(|w| w[0] <= w[1] + 1e-12));
        assert!(levels[0] > -4.0 * t && levels[35] < 4.0 * t);
        // Symmetric about zero, since the lattice is bipartite.
        for (low, high) in levels.iter().zip(levels.iter().rev()) {
            assert!(close(*low, -high, 1e-9), "the spectrum is not symmetric");
        }

        // Graphene: the two bands meet at the Dirac points and nowhere else
        // nearby.
        for &(kx, ky) in &dirac_points_graphene() {
            let (lower, upper) = graphene_dispersion(kx, ky, t);
            assert!(
                close(upper - lower, 0.0, 1e-9),
                "the gap at ({kx}, {ky}) is {}",
                upper - lower
            );
        }
        // Just away from a Dirac point the gap opens linearly, which is what
        // makes the carriers massless.
        let (kx, ky) = dirac_points_graphene()[0];
        let mut previous = 0.0;
        for delta in [0.005f64, 0.01, 0.02, 0.04] {
            let (lower, upper) = graphene_dispersion(kx + delta, ky, t);
            let gap = upper - lower;
            assert!(gap > previous, "the gap did not grow at delta = {delta}");
            if previous > 0.0 {
                let ratio = gap / previous;
                assert!(
                    (1.9..2.1).contains(&ratio),
                    "doubling the distance changed the gap by {ratio}, not linearly"
                );
            }
            previous = gap;
        }
        // At the zone centre the bands are as far apart as they get.
        let (lower, upper) = graphene_dispersion(0.0, 0.0, t);
        assert!(close(upper, 3.0 * t, 1e-9) && close(lower, -3.0 * t, 1e-9));

        assert!(tight_binding_square(0, 5, t).is_err());
        assert!(tight_binding_square(500, 500, t).is_err());
    }

    // -----------------------------------------------------------------
    // Kronig-Penney
    // -----------------------------------------------------------------

    #[test]
    fn the_kronig_penney_lattice_has_bands_that_widen_as_the_barrier_falls() {
        // With no barrier the spectrum is free and every energy is allowed;
        // as the barrier rises the bands narrow toward isolated levels. Both
        // limits are checked, and the monotone trend between them.
        let (a, b, mass, hbar) = (1.0f64, 0.3f64, 1.0f64, 1.0f64);
        let free = kronig_penney_bands(0.0, a, b, (0.01, 60.0), 40_000, mass, hbar).unwrap();
        let free_width: f64 = free.iter().map(|(lo, hi)| hi - lo).sum();
        assert!(
            free_width > 59.0,
            "with no barrier almost everything should be allowed, got {free_width}"
        );

        let mut previous = free_width;
        for v0 in [2.0f64, 10.0, 40.0, 150.0] {
            let bands = kronig_penney_bands(v0, a, b, (0.01, 60.0), 40_000, mass, hbar).unwrap();
            let width: f64 = bands.iter().map(|(lo, hi)| hi - lo).sum();
            assert!(
                width < previous,
                "raising the barrier to {v0} widened the allowed set to {width}"
            );
            assert!(!bands.is_empty(), "every barrier leaves some bands");
            // Inside a band the dispersion function is bounded by one.
            for (lo, hi) in &bands {
                let middle = 0.5 * (lo + hi);
                let value = kronig_penney(v0, a, b, middle, mass, hbar).unwrap();
                assert!(
                    value.abs() <= 1.0 + 1e-9,
                    "the middle of a band has |f| = {}",
                    value.abs()
                );
            }
            previous = width;
        }
        assert!(previous < free_width / 2.0, "a tall barrier should narrow the bands sharply");

        assert!(kronig_penney(1.0, 0.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(kronig_penney_bands(1.0, 1.0, 1.0, (2.0, 1.0), 100, 1.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Densities of states and occupations
    // -----------------------------------------------------------------

    #[test]
    fn the_free_electron_densities_of_states_have_the_dimensional_dependence_they_should() {
        // One over root E, constant, and root E. Each is checked by its
        // scaling with energy rather than by a single value, which is what
        // distinguishes them.
        let (m, hbar) = (1.0f64, 1.0f64);
        let one_a = density_of_states_1d_free(1.0, m, hbar).unwrap();
        let one_b = density_of_states_1d_free(4.0, m, hbar).unwrap();
        assert!(close(one_a / one_b, 2.0, 1e-9), "the one-dimensional ratio is {}", one_a / one_b);

        let two_a = density_of_states_2d_free(1.0, m, hbar).unwrap();
        let two_b = density_of_states_2d_free(9.0, m, hbar).unwrap();
        assert!(close(two_a, two_b, 1e-12), "the two-dimensional density is not constant");

        let three_a = density_of_states_3d_free(1.0, m, hbar).unwrap();
        let three_b = density_of_states_3d_free(4.0, m, hbar).unwrap();
        assert!(close(three_b / three_a, 2.0, 1e-9), "the three-dimensional ratio is wrong");

        // All vanish below the band bottom.
        for f in [
            density_of_states_1d_free as fn(f64, f64, f64) -> Result<f64, GeomError>,
            density_of_states_2d_free,
            density_of_states_3d_free,
        ] {
            assert_eq!(f(-1.0, m, hbar).unwrap(), 0.0);
            assert_eq!(f(0.0, m, hbar).unwrap(), 0.0);
            assert!(f(1.0, 0.0, hbar).is_err());
        }

        // The three-dimensional density integrates to the electron count
        // implied by the Fermi energy, which ties it to fermi_energy_free.
        let density = 8.5e28f64;
        let fermi = fermi_energy_free(density, ELECTRON_MASS).unwrap();
        let samples = 200_000usize;
        let h = fermi / samples as f64;
        let integral: f64 = (0..samples)
            .map(|k| {
                density_of_states_3d_free((k as f64 + 0.5) * h, ELECTRON_MASS, HBAR).unwrap()
            })
            .sum::<f64>()
            * h;
        assert!(
            relative(integral, density) < 1e-4,
            "the density of states integrates to {integral}, not {density}"
        );
        // Copper's Fermi energy is about seven electronvolts.
        assert!(
            (fermi / ELEMENTARY_CHARGE - 7.0).abs() < 0.5,
            "the Fermi energy is {} electronvolts",
            fermi / ELEMENTARY_CHARGE
        );
        assert!(fermi_energy_free(0.0, ELECTRON_MASS).is_err());
    }

    #[test]
    fn the_occupations_have_the_limits_and_symmetries_they_should() {
        let mu = 1.0e-19f64;
        // At zero temperature the Fermi function is a step.
        assert_eq!(fermi_dirac(mu * 0.5, mu, 0.0).unwrap(), 1.0);
        assert_eq!(fermi_dirac(mu * 1.5, mu, 0.0).unwrap(), 0.0);
        assert_eq!(fermi_dirac(mu, mu, 0.0).unwrap(), 0.5);
        // At any temperature it is one half at the chemical potential, and
        // antisymmetric about it.
        for t in [1.0f64, 300.0, 5000.0] {
            assert!(close(fermi_dirac(mu, mu, t).unwrap(), 0.5, 1e-15));
            for delta in [1e-21f64, 1e-20, 5e-20] {
                let above = fermi_dirac(mu + delta, mu, t).unwrap();
                let below = fermi_dirac(mu - delta, mu, t).unwrap();
                assert!(close(above + below, 1.0, 1e-12), "the function is not antisymmetric");
                assert!((0.0..=1.0).contains(&above));
            }
            // Far above the chemical potential it becomes Boltzmann, and it
            // does not overflow doing so.
            let far = fermi_dirac(mu + 100.0 * BOLTZMANN * t, mu, t).unwrap();
            assert!(far > 0.0 && far < 1e-40, "the tail is {far}");
        }

        // Bosons diverge as the energy approaches the chemical potential.
        let mut previous = 0.0;
        for delta in [1e-20f64, 1e-21, 1e-22] {
            let n = bose_einstein(mu + delta, mu, 300.0).unwrap();
            assert!(n > previous, "the occupation fell as the gap closed");
            previous = n;
        }
        // And at high temperature both tend to the classical count.
        let energy = mu + 1e-21;
        let classical = BOLTZMANN * 1e6 / (energy - mu);
        assert!(
            relative(bose_einstein(energy, mu, 1e6).unwrap(), classical) < 1e-3,
            "the classical limit fails"
        );
        assert!(bose_einstein(mu, mu, 300.0).is_err());
        assert!(bose_einstein(mu * 2.0, mu, -1.0).is_err());
        assert!(fermi_dirac(mu, mu, -1.0).is_err());
    }

    #[test]
    fn a_broadened_spectrum_integrates_to_the_number_of_levels_it_came_from() {
        let levels = [-2.0f64, -1.0, -1.0, 0.5, 3.0];
        let curve = dos_from_bands(&levels, 0.1, 4000).unwrap();
        let h = curve[1].0 - curve[0].0;
        let total: f64 = curve.iter().map(|(_, d)| d).sum::<f64>() * h;
        assert!(
            relative(total, levels.len() as f64) < 1e-3,
            "the density integrates to {total}, not {}",
            levels.len()
        );
        // The doubled level is twice as tall as a single one.
        let at = |e: f64| {
            curve
                .iter()
                .min_by(|a, b| {
                    (a.0 - e).abs().partial_cmp(&(b.0 - e).abs()).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
                .1
        };
        assert!(
            (at(-1.0) / at(3.0) - 2.0).abs() < 0.05,
            "the ratio is {}",
            at(-1.0) / at(3.0)
        );
        assert!(dos_from_bands(&[], 0.1, 100).is_err());
        assert!(dos_from_bands(&levels, 0.0, 100).is_err());
    }

    // -----------------------------------------------------------------
    // Heat capacities
    // -----------------------------------------------------------------

    #[test]
    fn debye_goes_as_t_cubed_at_low_temperature_and_to_dulong_petit_at_high() {
        let theta = 400.0f64;
        // High temperature: three k per atom.
        for t in [4000.0f64, 20_000.0] {
            let c = debye_heat_capacity(t, theta).unwrap();
            assert!(
                relative(c, 3.0 * BOLTZMANN) < 0.01,
                "at {t} kelvin the capacity is {} against {}",
                c,
                3.0 * BOLTZMANN
            );
        }
        // Low temperature: the cube law, checked by the ratio rather than the
        // constant.
        for pair in [(4.0f64, 8.0f64), (8.0, 16.0), (2.0, 4.0)] {
            let ratio =
                debye_heat_capacity(pair.1, theta).unwrap() / debye_heat_capacity(pair.0, theta).unwrap();
            assert!(
                (ratio - 8.0).abs() < 0.05,
                "doubling from {} to {} changed the capacity by {ratio}, not eight",
                pair.0,
                pair.1
            );
        }
        // And the coefficient matches the closed form 12 pi^4 k / 5 (T/theta)^3.
        let t = 4.0f64;
        let predicted = 12.0 * std::f64::consts::PI.powi(4) / 5.0 * BOLTZMANN * (t / theta).powi(3);
        assert!(
            relative(debye_heat_capacity(t, theta).unwrap(), predicted) < 0.01,
            "the low-temperature coefficient is off: {} against {predicted}",
            debye_heat_capacity(t, theta).unwrap()
        );
        assert_eq!(debye_heat_capacity(0.0, theta).unwrap(), 0.0);
        assert!(debye_heat_capacity(1.0, 0.0).is_err());

        // Einstein: the same high-temperature limit but an exponential
        // low-temperature fall, which is where the model is wrong.
        assert!(relative(einstein_heat_capacity(50_000.0, theta).unwrap(), 3.0 * BOLTZMANN) < 0.01);
        let low_einstein = einstein_heat_capacity(20.0, theta).unwrap();
        let low_debye = debye_heat_capacity(20.0, theta).unwrap();
        assert!(
            low_einstein < low_debye / 10.0,
            "Einstein should fall far faster: {low_einstein} against {low_debye}"
        );
        assert_eq!(einstein_heat_capacity(0.0, theta).unwrap(), 0.0);
        assert!(einstein_heat_capacity(1.0, -1.0).is_err());

        // Sommerfeld: linear, and tiny compared with the lattice at room
        // temperature -- which is the historical point.
        let fermi_temperature = 8.0e4f64;
        let electronic = sommerfeld_heat_capacity(300.0, fermi_temperature).unwrap();
        assert!(
            close(
                sommerfeld_heat_capacity(600.0, fermi_temperature).unwrap(),
                2.0 * electronic,
                1e-28
            ),
            "the electronic capacity is not linear"
        );
        assert!(
            electronic < 0.05 * 3.0 * BOLTZMANN,
            "the electronic capacity is {electronic}, not small against the lattice"
        );
        assert!(sommerfeld_heat_capacity(300.0, 0.0).is_err());
    }

    // -----------------------------------------------------------------
    // Phonons and fields
    // -----------------------------------------------------------------

    #[test]
    fn phonons_are_linear_at_long_wavelength_and_the_diatomic_chain_opens_a_gap() {
        let (spring, mass, a) = (4.0f64, 2.0f64, 1.0f64);
        let sound = (spring / mass).sqrt() * a;
        for k in [0.001f64, 0.002, 0.004] {
            let omega = phonon_dispersion_1d_monatomic(k, spring, mass, a);
            assert!(
                relative(omega, sound * k) < 1e-4,
                "at k = {k} the frequency is {omega}, the sound line {}",
                sound * k
            );
        }
        // The band top is at the zone boundary.
        let top = phonon_dispersion_1d_monatomic(std::f64::consts::PI / a, spring, mass, a);
        assert!(close(top, 2.0 * (spring / mass).sqrt(), 1e-12));
        assert!(close(phonon_dispersion_1d_monatomic(0.0, spring, mass, a), 0.0, 1e-15));

        // The diatomic chain: the acoustic branch still starts at zero, the
        // optical one does not, and they never cross.
        let (m1, m2) = (1.0f64, 3.0f64);
        let (acoustic0, optical0) = phonon_dispersion_1d_diatomic(0.0, spring, m1, m2, a);
        assert!(close(acoustic0, 0.0, 1e-12), "the acoustic branch starts at {acoustic0}");
        assert!(optical0 > 0.0, "the optical branch starts at zero");
        assert!(
            close(optical0, (2.0 * spring * (1.0 / m1 + 1.0 / m2)).sqrt(), 1e-9),
            "the optical branch at k = 0 is {optical0}"
        );
        for steps in 0..40usize {
            let k = std::f64::consts::PI / (2.0 * a) * steps as f64 / 40.0;
            let (acoustic, optical) = phonon_dispersion_1d_diatomic(k, spring, m1, m2, a);
            assert!(acoustic <= optical + 1e-12, "the branches crossed at k = {k}");
            assert!(acoustic >= 0.0 && optical >= 0.0);
        }
        // Equal masses close the gap: the diatomic chain becomes monatomic.
        let (a_eq, o_eq) = phonon_dispersion_1d_diatomic(
            std::f64::consts::PI / (2.0 * a),
            spring,
            mass,
            mass,
            a,
        );
        assert!(
            close(a_eq, o_eq, 1e-9),
            "equal masses should close the gap: {a_eq} against {o_eq}"
        );
    }

    #[test]
    fn landau_levels_are_equally_spaced_and_the_hall_conductance_is_quantised() {
        let field = 5.0f64;
        let spacing = HBAR * ELEMENTARY_CHARGE * field / ELECTRON_MASS;
        for n in 0..8usize {
            let level = landau_levels(field, n, ELECTRON_MASS).unwrap();
            // A level here is around 1e-22 joules, so the comparison has to
            // be relative: an absolute tolerance tighter than 1e-38 is below
            // what a double can represent at this magnitude.
            assert!(
                relative(level, (n as f64 + 0.5) * spacing) < 1e-12,
                "level {n} is {level}"
            );
            if n > 0 {
                let gap = level - landau_levels(field, n - 1, ELECTRON_MASS).unwrap();
                assert!(relative(gap, spacing) < 1e-12, "the spacing is {gap}");
            }
        }
        // The spacing is linear in the field and inverse in the mass.
        assert!(relative(
            landau_levels(10.0, 0, ELECTRON_MASS).unwrap(),
            2.0 * landau_levels(5.0, 0, ELECTRON_MASS).unwrap()
        ) < 1e-12);
        assert!(landau_levels(0.0, 1, ELECTRON_MASS).is_err());

        // The conductance quantum is e^2 / h, and the plateaux are its
        // integer multiples.
        let quantum = quantum_hall_conductance(1);
        assert!(
            relative(quantum, 3.874_045_86e-5) < 1e-6,
            "the conductance quantum is {quantum} siemens"
        );
        for n in 1..6usize {
            assert!(relative(quantum_hall_conductance(n), n as f64 * quantum) < 1e-12);
        }
        assert_eq!(quantum_hall_conductance(0), 0.0);
        // Its reciprocal is the von Klitzing constant, about 25.8 kilohms.
        assert!(relative(1.0 / quantum, 25_812.807) < 1e-6);
    }

    #[test]
    fn the_hofstadter_spectrum_splits_into_as_many_bands_as_the_flux_denominator() {
        // The band count is the *denominator* of the flux, which is why the
        // spectrum is nowhere continuous in the flux -- the defining feature
        // of the butterfly.
        let points = hofstadter_butterfly(6, 4).unwrap();
        assert!(!points.is_empty());
        // Every energy lies in the bandwidth of the square lattice.
        for (flux, energy) in &points {
            assert!((0.0..1.0).contains(flux), "the flux is {flux}");
            assert!(energy.abs() <= 4.5, "an energy of {energy} is outside the band");
        }
        // At half flux there are two sub-bands, and they are symmetric.
        let half: Vec<f64> = points
            .iter()
            .filter(|(flux, _)| close(*flux, 0.5, 1e-12))
            .map(|(_, e)| *e)
            .collect();
        assert!(!half.is_empty(), "half flux produced nothing");
        let positive = half.iter().filter(|e| **e > 1e-9).count();
        let negative = half.iter().filter(|e| **e < -1e-9).count();
        assert_eq!(positive, negative, "the half-flux spectrum is not symmetric");
        assert_eq!(positive * 2, half.len(), "half flux should have two sub-bands");

        // At flux one third there are three, and their count is what the
        // denominator says.
        for (p, q) in [(1usize, 3usize), (1, 4), (2, 5)] {
            let at: Vec<f64> = points
                .iter()
                .filter(|(flux, _)| close(*flux, p as f64 / q as f64, 1e-12))
                .map(|(_, e)| *e)
                .collect();
            assert_eq!(
                at.len() % q,
                0,
                "flux {p}/{q} gave {} energies, not a multiple of {q}",
                at.len()
            );
        }
        assert!(hofstadter_butterfly(1, 4).is_err());
        assert!(hofstadter_butterfly(6, 0).is_err());
    }

    #[test]
    fn the_effective_mass_is_positive_at_a_band_bottom_and_negative_at_the_top() {
        // The negative mass at the band top is not a curiosity: it is what a
        // hole is, and the reason a nearly full band conducts as though its
        // carriers were positive.
        let t = 1.0e-19f64;
        let a = 3.0e-10f64;
        let band = |k: f64| tight_binding_band_1d(k, t, a);
        let bottom = effective_mass_from_band(&band, 0.0, 1e-4 / a).unwrap();
        let expected = HBAR * HBAR / (2.0 * t * a * a);
        assert!(
            relative(bottom, expected) < 1e-4,
            "the band-bottom mass is {bottom}, the closed form {expected}"
        );
        assert!(bottom > 0.0);

        let top = effective_mass_from_band(&band, std::f64::consts::PI / a, 1e-4 / a).unwrap();
        assert!(top < 0.0, "the band-top mass is {top}, not negative");
        assert!(relative(top.abs(), expected) < 1e-4);

        // A free-electron band gives the free mass exactly.
        let free = |k: f64| HBAR * HBAR * k * k / (2.0 * ELECTRON_MASS);
        let mass = effective_mass_from_band(&free, 1e9, 1e6).unwrap();
        assert!(relative(mass, ELECTRON_MASS) < 1e-6, "the free mass came out {mass}");

        assert!(effective_mass_from_band(&|_| 1.0, 0.0, 1e-3).is_err());
        assert!(effective_mass_from_band(&band, 0.0, 0.0).is_err());
    }

    #[test]
    fn a_bloch_oscillation_is_faster_in_a_stronger_field_and_a_wider_lattice() {
        let period = bloch_oscillation_period(1e5, 1e-8).unwrap();
        assert!(close(bloch_oscillation_period(2e5, 1e-8).unwrap(), period / 2.0, 1e-20));
        assert!(close(bloch_oscillation_period(1e5, 2e-8).unwrap(), period / 2.0, 1e-20));
        // In an ordinary crystal at an ordinary field the period is far longer
        // than any scattering time, which is why it is never seen there.
        let ordinary = bloch_oscillation_period(1e5, 3e-10).unwrap();
        assert!(
            ordinary > 1e-10,
            "the period is {ordinary} seconds, shorter than a scattering time"
        );
        assert!(bloch_oscillation_period(0.0, 1e-9).is_err());
        assert!(bloch_oscillation_period(1e5, 0.0).is_err());
    }

    // -----------------------------------------------------------------
    // Transport, semiconductors, superconductors
    // -----------------------------------------------------------------

    #[test]
    fn drude_and_landauer_give_the_conductances_they_promise() {
        let density = 8.5e28f64;
        let sigma = drude_conductivity(density, 2.5e-14, ELECTRON_MASS).unwrap();
        // Copper's conductivity is about 6 x 10^7 siemens per metre.
        assert!(
            (sigma / 6.0e7 - 1.0).abs() < 0.2,
            "the conductivity is {sigma} siemens per metre"
        );
        // Linear in the density and in the scattering time.
        assert!(close(
            drude_conductivity(2.0 * density, 2.5e-14, ELECTRON_MASS).unwrap(),
            2.0 * sigma,
            1e-6 * sigma
        ));
        assert!(drude_conductivity(density, 0.0, ELECTRON_MASS).is_err());

        // The Hall coefficient's sign says which carrier moves.
        assert!(hall_coefficient(density, -ELEMENTARY_CHARGE).unwrap() < 0.0);
        assert!(hall_coefficient(density, ELEMENTARY_CHARGE).unwrap() > 0.0);
        assert!(hall_coefficient(0.0, ELEMENTARY_CHARGE).is_err());

        // Landauer: even a perfect channel has finite conductance.
        let quantum = conductance_landauer(&[1.0]).unwrap();
        assert!(
            relative(quantum, 7.748_091_729e-5) < 1e-6,
            "the conductance quantum is {quantum}"
        );
        assert!(relative(conductance_landauer(&[1.0; 4]).unwrap(), 4.0 * quantum) < 1e-12);
        assert!(relative(conductance_landauer(&[0.5, 0.5]).unwrap(), quantum) < 1e-12);
        assert_eq!(conductance_landauer(&[]).unwrap(), 0.0);
        assert!(conductance_landauer(&[1.2]).is_err());
        assert!(conductance_landauer(&[-0.1]).is_err());
    }

    #[test]
    fn a_semiconductor_and_its_junction_behave_as_the_exponentials_say() {
        // Silicon at room temperature has about 10^16 carriers per cubic
        // metre, and the density doubles for roughly every eight kelvin.
        let n300 = semiconductor_carrier_density(1.12, 300.0, 1.08, 0.81).unwrap();
        assert!(
            (n300.log10() - 16.0).abs() < 0.6,
            "the intrinsic density is 10^{} per cubic metre",
            n300.log10()
        );
        let n308 = semiconductor_carrier_density(1.12, 308.0, 1.08, 0.81).unwrap();
        let ratio = n308 / n300;
        assert!((1.8..2.6).contains(&ratio), "eight kelvin changed it by {ratio}");
        // A wider gap means fewer carriers, sharply.
        let wide = semiconductor_carrier_density(3.3, 300.0, 1.0, 1.0).unwrap();
        assert!(wide < n300 * 1e-15, "gallium nitride should be far more insulating");
        assert!(semiconductor_carrier_density(1.1, 0.0, 1.0, 1.0).is_err());

        // The junction's built-in potential is a fraction of the gap and
        // grows logarithmically with the doping.
        let v0 = pn_junction_builtin(1e22, 1e22, n300, 300.0).unwrap();
        assert!((0.3..1.12).contains(&v0), "the built-in potential is {v0} volts");
        let heavier = pn_junction_builtin(1e24, 1e24, n300, 300.0).unwrap();
        assert!(heavier > v0, "heavier doping should raise the barrier");
        let thermal = BOLTZMANN * 300.0 / ELEMENTARY_CHARGE;
        assert!(
            close(heavier - v0, thermal * (1e4f64).ln(), 1e-9),
            "the increase is {} volts",
            heavier - v0
        );

        // The depletion width shrinks as the doping rises.
        let wide_w = depletion_width(v0, 1e21, 1e21, 11.7).unwrap();
        let narrow_w = depletion_width(v0, 1e24, 1e24, 11.7).unwrap();
        assert!(narrow_w < wide_w, "heavier doping should narrow the depletion region");
        assert!(
            (wide_w / narrow_w - (1e3f64).sqrt()).abs() < 1e-6 * (1e3f64).sqrt(),
            "the width should go as the inverse square root of the doping"
        );
        assert!((1e-8..1e-5).contains(&wide_w), "the width is {wide_w} metres");
        assert!(pn_junction_builtin(0.0, 1e22, 1e16, 300.0).is_err());
        assert!(depletion_width(1.0, 0.0, 1e22, 11.7).is_err());
    }

    #[test]
    fn the_superconducting_gap_opens_as_a_square_root_and_the_josephson_relations_hold() {
        let tc = 9.3f64;
        assert!(close(bcs_gap_equation(0.0, tc).unwrap(), 1.0, 1e-12));
        assert_eq!(bcs_gap_equation(tc, tc).unwrap(), 0.0);
        assert_eq!(bcs_gap_equation(2.0 * tc, tc).unwrap(), 0.0);
        // Monotone in temperature.
        let mut previous = 1.1;
        for t in [0.1f64, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let gap = bcs_gap_equation(t * tc, tc).unwrap();
            assert!(gap < previous, "the gap rose at t = {t}");
            assert!((0.0..=1.0).contains(&gap));
            previous = gap;
        }
        // Just below Tc it opens as sqrt(1 - T / Tc).
        let ratio = bcs_gap_equation(0.99 * tc, tc).unwrap()
            / bcs_gap_equation(0.9999 * tc, tc).unwrap();
        assert!((9.0..11.0).contains(&ratio), "the opening ratio is {ratio}, not near ten");
        assert!(bcs_gap_equation(1.0, 0.0).is_err());

        // The critical temperature is exponentially small in the coupling.
        let weak = bcs_tc_from_coupling(0.2, 400.0).unwrap();
        let strong = bcs_tc_from_coupling(0.4, 400.0).unwrap();
        assert!(strong > 10.0 * weak, "doubling the coupling gave {strong} against {weak}");
        assert!(close(
            bcs_tc_from_coupling(0.3, 800.0).unwrap(),
            2.0 * bcs_tc_from_coupling(0.3, 400.0).unwrap(),
            1e-9
        ));
        assert!(bcs_tc_from_coupling(0.0, 400.0).is_err());

        // Josephson: a supercurrent at zero voltage, and 484 megahertz per
        // microvolt.
        assert!(close(josephson_current(1e-6, 0.0), 0.0, 1e-18));
        assert!(relative(josephson_current(1e-6, std::f64::consts::FRAC_PI_2), 1e-6) < 1e-12);
        assert!(close(josephson_current(1e-6, std::f64::consts::PI), 0.0, 1e-18));
        assert!(
            relative(josephson_frequency(1e-6), 483.597_848_4e6) < 1e-6,
            "the frequency is {} hertz per microvolt",
            josephson_frequency(1e-6)
        );
        assert!(close(josephson_frequency(0.0), 0.0, 1e-12));
    }

    #[test]
    fn every_state_of_a_disordered_chain_is_localised_and_more_so_at_stronger_disorder() {
        // The one-dimensional result is absolute: any disorder localises
        // everything. What varies is the length, which shrinks as the
        // disorder grows.
        let mut rng = Rng::new(0x_5011_0001);
        let mut previous = f64::INFINITY;
        for disorder in [0.5f64, 1.0, 2.0, 4.0, 8.0] {
            let length = anderson_localization_1d(4000, disorder, 0.0, 20, &mut rng).unwrap();
            assert!(length > 0.0 && length.is_finite(), "the length is {length}");
            assert!(
                length < previous,
                "raising the disorder to {disorder} lengthened the states to {length}"
            );
            previous = length;
        }
        assert!(previous < 2.0, "strong disorder should localise within a few sites: {previous}");

        // At weak disorder the length goes as the inverse square of it,
        // which is the perturbative result. The chain has to be far longer
        // than the length being measured for the Lyapunov exponent to
        // self-average: at W = 0.2 the states run to some three thousand
        // sites, and a twenty-thousand-site chain gives only seven of them,
        // which leaves enough scatter in the ratio to invent effects that
        // are not there.
        let mut fresh = Rng::new(0x_5011_0002);
        for energy in [0.0f64, 0.5] {
            let weak = anderson_localization_1d(150_000, 0.4, energy, 24, &mut fresh).unwrap();
            let weaker = anderson_localization_1d(150_000, 0.2, energy, 24, &mut fresh).unwrap();
            let ratio = weaker / weak;
            // The tolerance is set by the estimator's own scatter, which at
            // this chain length and trial count leaves a few per cent on the
            // Lyapunov exponent and rather more on the ratio of two of them.
            assert!(
                (3.6..4.4).contains(&ratio),
                "at E = {energy}, halving the disorder changed the length by {ratio}"
            );
            // And the lengths themselves are long compared with the lattice
            // but short compared with the chain, which is the regime where
            // the measurement means anything.
            assert!(
                (100.0..20_000.0).contains(&weaker),
                "the weak-disorder length is {weaker} sites"
            );
        }

        assert!(anderson_localization_1d(5, 1.0, 0.0, 10, &mut rng).is_err());
        assert!(anderson_localization_1d(100, 0.0, 0.0, 10, &mut rng).is_err());
        assert!(anderson_localization_1d(100, 1.0, 0.0, 0, &mut rng).is_err());
    }
}
