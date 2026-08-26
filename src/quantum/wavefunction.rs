//! One-dimensional wavefunctions, the standard eigenstates, and phase-space
//! distributions.
//!
//! Everything here works in whatever unit system the caller supplies through
//! `hbar` and the masses, so the natural choice for testing -- `hbar = m = 1`
//! -- is available alongside SI. That matters more than it sounds: the
//! quantities that can be checked exactly, like the harmonic oscillator's
//! `(n + 1/2) hbar omega` spectrum or a Gaussian's saturation of the
//! uncertainty bound, are clearest when the constants are one, and a module
//! that hard-codes SI cannot express them.
//!
//! The one thing worth stating up front is the discretisation. A wavefunction
//! is represented by its samples on a uniform grid, and every integral below
//! is the corresponding Riemann sum. That is exact for none of them and
//! spectrally accurate for a smooth function that has decayed to nothing at
//! both ends -- which is the condition the callers here are responsible for
//! arranging, and the one under which the tests hold to the tolerances they
//! state.

use crate::error::GeomError;
use crate::fractals::Complex;
use crate::special::legendre::spherical_harmonic_real;
use crate::transforms::fft::{fft, ifft};

/// A complex wavefunction sampled on a uniform grid.
///
/// The grid runs from `x0` in steps of `dx`, so sample `k` sits at
/// `x0 + k * dx`.
#[derive(Debug, Clone)]
pub struct Wavefunction1D {
    /// The samples.
    pub psi: Vec<Complex>,
    /// Grid spacing.
    pub dx: f64,
    /// Position of the first sample.
    pub x0: f64,
}

fn scale(z: Complex, k: f64) -> Complex {
    Complex::new(z.re * k, z.im * k)
}

impl Wavefunction1D {
    /// A wavefunction from explicit samples.
    ///
    /// # Errors
    /// Returns an error for an empty sample vector or a non-positive spacing.
    pub fn new(psi: Vec<Complex>, dx: f64, x0: f64) -> Result<Self, GeomError> {
        if psi.is_empty() {
            return Err(GeomError::InvalidArgument("a wavefunction needs samples"));
        }
        if !(dx > 0.0) {
            return Err(GeomError::InvalidArgument("the grid spacing must be positive"));
        }
        Ok(Self { psi, dx, x0 })
    }

    /// A normalised Gaussian wave packet centred at `centre` with mean
    /// momentum `hbar * k0` and position spread `sigma`.
    ///
    /// The minimum-uncertainty state: it saturates `sigma_x sigma_p = hbar/2`
    /// exactly, and it is the only state that does. Everything else in
    /// quantum mechanics has a strictly larger product, so this is the
    /// reference against which "how close to classical" is measured.
    ///
    /// # Errors
    /// Returns an error for a non-positive width or an empty grid.
    pub fn gaussian_packet(
        centre: f64,
        k0: f64,
        sigma: f64,
        dx: f64,
        x0: f64,
        n: usize,
    ) -> Result<Self, GeomError> {
        if !(sigma > 0.0) {
            return Err(GeomError::InvalidArgument("the packet width must be positive"));
        }
        let psi: Vec<Complex> = (0..n)
            .map(|k| {
                let x = x0 + k as f64 * dx;
                let gaussian = (-(x - centre) * (x - centre) / (4.0 * sigma * sigma)).exp();
                let phase = k0 * x;
                Complex::new(gaussian * phase.cos(), gaussian * phase.sin())
            })
            .collect();
        let mut w = Self::new(psi, dx, x0)?;
        w.normalize();
        Ok(w)
    }

    /// A plane wave `exp(i k x)` on the grid, normalised over it.
    ///
    /// Not normalisable on the whole line -- which is why momentum
    /// eigenstates are not states -- so this is the box-normalised stand-in.
    ///
    /// # Errors
    /// Returns an error for an empty grid or a non-positive spacing.
    pub fn plane_wave(k: f64, dx: f64, x0: f64, n: usize) -> Result<Self, GeomError> {
        let psi: Vec<Complex> = (0..n)
            .map(|j| {
                let phase = k * (x0 + j as f64 * dx);
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();
        let mut w = Self::new(psi, dx, x0)?;
        w.normalize();
        Ok(w)
    }

    /// The number of grid points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.psi.len()
    }

    /// Always false: a wavefunction cannot be constructed empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The position of sample `k`.
    #[must_use]
    pub fn x(&self, k: usize) -> f64 {
        self.x0 + k as f64 * self.dx
    }

    /// `sqrt(integral |psi|^2 dx)` on the grid.
    #[must_use]
    pub fn norm(&self) -> f64 {
        (self.psi.iter().map(|z| z.norm_sq()).sum::<f64>() * self.dx).sqrt()
    }

    /// Scales the wavefunction to unit norm, leaving it alone if it is zero.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 0.0 {
            let inverse = 1.0 / n;
            for z in &mut self.psi {
                *z = scale(*z, inverse);
            }
        }
    }

    /// The probability density `|psi|^2`.
    #[must_use]
    pub fn probability_density(&self) -> Vec<f64> {
        self.psi.iter().map(|z| z.norm_sq()).collect()
    }

    /// The expected position.
    #[must_use]
    pub fn expectation_x(&self) -> f64 {
        let density = self.probability_density();
        let total: f64 = density.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        density.iter().enumerate().map(|(k, p)| p * self.x(k)).sum::<f64>() / total
    }

    /// The variance of position.
    #[must_use]
    pub fn variance_x(&self) -> f64 {
        let mean = self.expectation_x();
        let density = self.probability_density();
        let total: f64 = density.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        density
            .iter()
            .enumerate()
            .map(|(k, p)| p * (self.x(k) - mean) * (self.x(k) - mean))
            .sum::<f64>()
            / total
    }

    /// The grid's momentum values in FFT order, in units where the wavenumber
    /// is `k` and the momentum `hbar k`.
    ///
    /// The second half of the array holds the negative frequencies, which is
    /// the convention the FFT imposes and the one place a sign error hides
    /// most easily.
    #[must_use]
    pub fn wavenumbers(&self) -> Vec<f64> {
        let n = self.len();
        let span = n as f64 * self.dx;
        (0..n)
            .map(|k| {
                let index = if k <= n / 2 { k as f64 } else { k as f64 - n as f64 };
                2.0 * std::f64::consts::PI * index / span
            })
            .collect()
    }

    /// The momentum-space amplitudes, in FFT order.
    ///
    /// # Errors
    /// Returns an error unless the grid length is a power of two.
    pub fn momentum_space(&self) -> Result<Vec<Complex>, GeomError> {
        if !self.len().is_power_of_two() {
            return Err(GeomError::InvalidArgument("momentum_space needs a power-of-two grid"));
        }
        Ok(fft(&self.psi))
    }

    /// The expected momentum, in units of `hbar`.
    ///
    /// Computed spectrally rather than by differencing: the momentum operator
    /// is exactly diagonal in the Fourier basis, so on a periodic grid this is
    /// exact to rounding, while a finite difference carries an `O(dx^2)`
    /// error that then contaminates the uncertainty product.
    ///
    /// # Errors
    /// Returns an error unless the grid length is a power of two.
    pub fn expectation_k(&self) -> Result<f64, GeomError> {
        let spectrum = self.momentum_space()?;
        let weights: Vec<f64> = spectrum.iter().map(|z| z.norm_sq()).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Ok(0.0);
        }
        let k = self.wavenumbers();
        Ok(weights.iter().zip(&k).map(|(w, ki)| w * ki).sum::<f64>() / total)
    }

    /// The variance of the wavenumber.
    ///
    /// # Errors
    /// Returns an error unless the grid length is a power of two.
    pub fn variance_k(&self) -> Result<f64, GeomError> {
        let spectrum = self.momentum_space()?;
        let weights: Vec<f64> = spectrum.iter().map(|z| z.norm_sq()).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Ok(0.0);
        }
        let k = self.wavenumbers();
        let mean = self.expectation_k()?;
        Ok(weights
            .iter()
            .zip(&k)
            .map(|(w, ki)| w * (ki - mean) * (ki - mean))
            .sum::<f64>()
            / total)
    }

    /// The uncertainty product `sigma_x sigma_p` with the given `hbar`.
    ///
    /// Bounded below by `hbar / 2`, with equality exactly for a Gaussian.
    ///
    /// # Errors
    /// Returns an error unless the grid length is a power of two.
    pub fn uncertainty_product(&self, hbar: f64) -> Result<f64, GeomError> {
        Ok(self.variance_x().sqrt() * (hbar * self.variance_k()?.sqrt()))
    }

    /// The overlap `<other | self>`.
    ///
    /// # Errors
    /// Returns an error if the two grids disagree.
    pub fn overlap(&self, other: &Self) -> Result<Complex, GeomError> {
        if self.len() != other.len() || (self.dx - other.dx).abs() > 1e-15 {
            return Err(GeomError::InvalidArgument("overlap requires the same grid"));
        }
        let mut acc = Complex::new(0.0, 0.0);
        for (a, b) in other.psi.iter().zip(&self.psi) {
            acc = acc + a.conjugate() * *b;
        }
        Ok(scale(acc, self.dx))
    }

    /// The expected energy for the potential `v`, with the kinetic term
    /// evaluated spectrally.
    ///
    /// # Errors
    /// Returns an error if the potential has the wrong length or the grid is
    /// not a power of two.
    pub fn energy(&self, v: &[f64], hbar: f64, mass: f64) -> Result<f64, GeomError> {
        if v.len() != self.len() {
            return Err(GeomError::InvalidArgument("the potential has the wrong length"));
        }
        if !(mass > 0.0) {
            return Err(GeomError::InvalidArgument("the mass must be positive"));
        }
        let spectrum = self.momentum_space()?;
        let k = self.wavenumbers();
        let n = self.len() as f64;
        // Parseval on this FFT convention: sum |psi_hat|^2 = n sum |psi|^2.
        let kinetic: f64 = spectrum
            .iter()
            .zip(&k)
            .map(|(z, ki)| z.norm_sq() * hbar * hbar * ki * ki / (2.0 * mass))
            .sum::<f64>()
            * self.dx
            / n;
        let potential: f64 =
            self.psi.iter().zip(v).map(|(z, vi)| z.norm_sq() * vi).sum::<f64>() * self.dx;
        let weight = self.norm().powi(2);
        if weight <= 0.0 {
            return Ok(0.0);
        }
        Ok((kinetic + potential) / weight)
    }

    /// Applies the free-particle propagator for a time `t` spectrally.
    ///
    /// Exact for the free particle at any step size, since the kinetic
    /// operator is diagonal in momentum -- there is no time-stepping error to
    /// accumulate. That makes it the reference a split-operator integrator
    /// should be measured against.
    ///
    /// # Errors
    /// Returns an error unless the grid length is a power of two.
    pub fn propagate_free(&self, t: f64, hbar: f64, mass: f64) -> Result<Self, GeomError> {
        if !(mass > 0.0) {
            return Err(GeomError::InvalidArgument("the mass must be positive"));
        }
        let mut spectrum = self.momentum_space()?;
        let k = self.wavenumbers();
        for (z, ki) in spectrum.iter_mut().zip(&k) {
            let phase = -hbar * ki * ki * t / (2.0 * mass);
            *z = *z * Complex::new(phase.cos(), phase.sin());
        }
        Ok(Self { psi: ifft(&spectrum), dx: self.dx, x0: self.x0 })
    }
}

// ---------------------------------------------------------------------------
// Orthogonal polynomials
// ---------------------------------------------------------------------------

/// The physicists' Hermite polynomial `H_n(x)`.
///
/// Evaluated by the upward recurrence `H_{n+1} = 2x H_n - 2n H_{n-1}` rather
/// than from the explicit sum, whose alternating terms cancel catastrophically:
/// at `n = 20` and moderate `x` the largest term exceeds the answer by many
/// orders of magnitude, and a direct sum loses every significant digit.
#[must_use]
pub fn hermite_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut previous = 1.0;
    let mut current = 2.0 * x;
    for k in 1..n {
        let next = 2.0 * x * current - 2.0 * k as f64 * previous;
        previous = current;
        current = next;
    }
    current
}

/// The associated Laguerre polynomial `L_n^k(x)`.
///
/// Also by recurrence, and for the same reason.
#[must_use]
pub fn laguerre_associated(n: usize, k: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut previous = 1.0;
    let mut current = 1.0 + k - x;
    for j in 1..n {
        let jf = j as f64;
        let next = ((2.0 * jf + 1.0 + k - x) * current - (jf + k) * previous) / (jf + 1.0);
        previous = current;
        current = next;
    }
    current
}

// ---------------------------------------------------------------------------
// The standard eigenstates
// ---------------------------------------------------------------------------

/// The `n`-th harmonic oscillator eigenstate, normalised on the whole line.
///
/// The normalisation `(m omega / pi hbar)^(1/4) / sqrt(2^n n!)` is folded in
/// through logarithms, since `2^n n!` overflows a double at `n = 170` while
/// the state itself stays perfectly ordinary.
///
/// # Panics
/// Panics unless the mass, frequency and `hbar` are positive.
#[must_use]
pub fn harmonic_oscillator_eigenstate(
    n: usize,
    x: f64,
    mass: f64,
    omega: f64,
    hbar: f64,
) -> f64 {
    assert!(mass > 0.0 && omega > 0.0 && hbar > 0.0, "the oscillator parameters must be positive");
    let alpha = mass * omega / hbar;
    let xi = alpha.sqrt() * x;
    let mut log_norm = 0.25 * (alpha / std::f64::consts::PI).ln() - 0.5 * n as f64 * 2.0f64.ln();
    for k in 1..=n {
        log_norm -= 0.5 * (k as f64).ln();
    }
    log_norm.exp() * hermite_polynomial(n, xi) * (-0.5 * xi * xi).exp()
}

/// The energy of the `n`-th harmonic oscillator level: `(n + 1/2) hbar omega`.
///
/// The half is the zero-point energy, and it is not a convention: the ground
/// state cannot sit at the bottom of the well without violating the
/// uncertainty relation, and `hbar omega / 2` is exactly what the relation
/// costs.
///
/// # Panics
/// Panics unless `omega` and `hbar` are positive.
#[must_use]
pub fn harmonic_oscillator_energy(n: usize, omega: f64, hbar: f64) -> f64 {
    assert!(omega > 0.0 && hbar > 0.0, "the oscillator parameters must be positive");
    (n as f64 + 0.5) * hbar * omega
}

/// The `n`-th eigenstate of an infinite square well of width `l`, indexed
/// from one, and zero outside the well.
///
/// # Panics
/// Panics unless `n >= 1` and the width is positive.
#[must_use]
pub fn infinite_well_eigenstate(n: usize, x: f64, l: f64) -> f64 {
    assert!(n >= 1, "the well's states are indexed from one");
    assert!(l > 0.0, "the well must have a positive width");
    if x <= 0.0 || x >= l {
        return 0.0;
    }
    (2.0 / l).sqrt() * (n as f64 * std::f64::consts::PI * x / l).sin()
}

/// The energy of the `n`-th infinite-well level.
///
/// # Panics
/// Panics unless `n >= 1` and the width, mass and `hbar` are positive.
#[must_use]
pub fn infinite_well_energy(n: usize, l: f64, mass: f64, hbar: f64) -> f64 {
    assert!(n >= 1, "the well's states are indexed from one");
    assert!(l > 0.0 && mass > 0.0 && hbar > 0.0, "the well parameters must be positive");
    let k = n as f64 * std::f64::consts::PI / l;
    hbar * hbar * k * k / (2.0 * mass)
}

/// The hydrogen radial wavefunction `R_{n,l}(r)` in units of the Bohr radius
/// `a0`.
///
/// # Panics
/// Panics unless `n >= 1`, `l < n` and `a0` is positive.
#[must_use]
pub fn hydrogen_radial(n: usize, l: usize, r: f64, a0: f64) -> f64 {
    assert!(n >= 1 && l < n, "hydrogen states require 1 <= n and l < n");
    assert!(a0 > 0.0, "the Bohr radius must be positive");
    let rho = 2.0 * r / (n as f64 * a0);
    // The normalisation involves (n - l - 1)! and (n + l)!, which are taken
    // in logarithms so that large n does not overflow on the way to a small
    // number.
    let mut log_norm = 1.5 * (2.0 / (n as f64 * a0)).ln();
    let mut log_ratio = 0.0;
    for k in 1..=(n - l - 1) {
        log_ratio += (k as f64).ln();
    }
    for k in 1..=(n + l) {
        log_ratio -= (k as f64).ln();
    }
    log_norm += 0.5 * (log_ratio - (2.0 * n as f64).ln());
    log_norm.exp()
        * (-rho / 2.0).exp()
        * rho.powi(l as i32)
        * laguerre_associated(n - l - 1, 2.0 * l as f64 + 1.0, rho)
}

/// The hydrogen energy level in electronvolts: `-13.6 / n^2`.
///
/// # Panics
/// Panics unless `n >= 1`.
#[must_use]
pub fn hydrogen_energy(n: usize) -> f64 {
    assert!(n >= 1, "hydrogen levels are indexed from one");
    -13.605_693_122_994 / (n * n) as f64
}

/// The probability density of a real hydrogen orbital at a point in spherical
/// coordinates.
///
/// Uses the real spherical harmonics, so `m` selects the real combinations
/// -- the `p_x`, `p_y`, `p_z` shapes rather than the complex `m` eigenstates.
/// The two bases span the same space and give the same total density in a
/// shell; they differ in the angular shape of an individual orbital, which is
/// exactly what chemistry draws.
///
/// # Panics
/// Panics unless `n >= 1`, `l < n`, `|m| <= l` and `a0` is positive.
#[must_use]
pub fn hydrogen_orbital_density(
    n: usize,
    l: usize,
    m: i32,
    r: f64,
    theta: f64,
    phi: f64,
    a0: f64,
) -> f64 {
    assert!(m.unsigned_abs() as usize <= l, "hydrogen orbitals require |m| <= l");
    let radial = hydrogen_radial(n, l, r, a0);
    let angular = spherical_harmonic_real(l as u32, m, theta, phi);
    radial * radial * angular * angular
}

// ---------------------------------------------------------------------------
// Fock-space states
// ---------------------------------------------------------------------------

/// The Fock coefficients of a coherent state `|alpha>`, truncated at
/// `n_max` photons.
///
/// A Poisson distribution over photon number with mean `|alpha|^2`. Coherent
/// states are the eigenstates of the annihilation operator, which is why
/// removing a photon from a laser beam leaves it unchanged, and why the
/// photon statistics of a laser are Poissonian rather than thermal.
///
/// # Errors
/// Returns an error for an empty truncation.
pub fn coherent_state(alpha: Complex, n_max: usize) -> Result<Vec<Complex>, GeomError> {
    if n_max == 0 {
        return Err(GeomError::InvalidArgument("coherent_state needs a positive truncation"));
    }
    let magnitude = alpha.norm();
    let phase = alpha.arg();
    let mut out = Vec::with_capacity(n_max);
    let mut log_term = -0.5 * magnitude * magnitude;
    for n in 0..n_max {
        if n > 0 {
            // alpha^n / sqrt(n!), carried in logarithms.
            log_term += magnitude.ln() - 0.5 * (n as f64).ln();
        }
        let weight = log_term.exp();
        let angle = phase * n as f64;
        out.push(Complex::new(weight * angle.cos(), weight * angle.sin()));
    }
    Ok(out)
}

/// The Fock coefficients of a squeezed vacuum state, truncated at `n_max`.
///
/// Only the even photon numbers are populated, because the squeezing operator
/// creates photons in pairs. That parity is the state's signature and is what
/// makes it useful: the noise removed from one quadrature has to go somewhere,
/// and it goes into the other.
///
/// # Errors
/// Returns an error for an empty truncation.
pub fn squeezed_state(r: f64, phi: f64, n_max: usize) -> Result<Vec<Complex>, GeomError> {
    if n_max == 0 {
        return Err(GeomError::InvalidArgument("squeezed_state needs a positive truncation"));
    }
    let mut out = vec![Complex::new(0.0, 0.0); n_max];
    let sech = 1.0 / r.cosh();
    let tanh = r.tanh();
    // c_{2k} = sqrt((2k)!) / (2^k k!) * (-e^{i phi} tanh r)^k * sqrt(sech r).
    let mut log_coefficient = 0.5 * sech.ln();
    for k in 0..n_max.div_ceil(2) {
        if k > 0 {
            let kf = k as f64;
            // The ratio of successive prefactors, in logarithms.
            log_coefficient +=
                0.5 * ((2.0 * kf - 1.0).ln() + (2.0 * kf).ln()) - kf.ln() - 2.0f64.ln();
            if tanh > 0.0 {
                log_coefficient += tanh.ln();
            } else {
                return Ok(out);
            }
        }
        let magnitude = log_coefficient.exp();
        // Each factor carries a minus sign and a phase.
        let angle = phi * k as f64 + std::f64::consts::PI * k as f64;
        out[2 * k] = Complex::new(magnitude * angle.cos(), magnitude * angle.sin());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase-space distributions
// ---------------------------------------------------------------------------

/// The Wigner function of a wavefunction at a point of phase space.
///
/// `W(x, p) = (1 / pi hbar) integral psi*(x + y) psi(x - y) e^{2 i p y / hbar} dy`.
///
/// The nearest thing quantum mechanics has to a phase-space probability
/// density: its marginals are the true position and momentum distributions.
/// It is not a probability density, because it takes negative values -- and
/// where it does is exactly where the state has no classical description, so
/// the negativity is the useful part rather than a defect of the definition.
///
/// # Errors
/// Returns an error for a non-positive spacing or `hbar`.
pub fn wigner_function(
    psi: &[Complex],
    dx: f64,
    x0: f64,
    x: f64,
    p: f64,
    hbar: f64,
) -> Result<f64, GeomError> {
    if psi.is_empty() || !(dx > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("wigner_function: bad grid"));
    }
    let n = psi.len();
    let centre = (x - x0) / dx;
    // The integral runs over the offsets for which both x + y and x - y stay
    // on the grid, which is what keeps the marginals right at the edges.
    let reach = centre.min(n as f64 - 1.0 - centre).floor().max(0.0) as usize;
    let mut acc = 0.0;
    for offset in 0..=reach {
        for sign in [1i64, -1] {
            if offset == 0 && sign < 0 {
                continue;
            }
            let step = sign * offset as i64;
            let plus = centre.round() as i64 + step;
            let minus = centre.round() as i64 - step;
            if plus < 0 || minus < 0 || plus >= n as i64 || minus >= n as i64 {
                continue;
            }
            let y = step as f64 * dx;
            let product = psi[plus as usize].conjugate() * psi[minus as usize];
            let angle = 2.0 * p * y / hbar;
            acc += product.re * angle.cos() - product.im * angle.sin();
        }
    }
    Ok(acc * dx / (std::f64::consts::PI * hbar))
}

/// The Husimi Q function: the Wigner function smoothed by a coherent state of
/// width `sigma`.
///
/// Smoothing over a phase-space cell of the minimum allowed area is exactly
/// enough to remove the negativity, so `Q` is a genuine probability density.
/// What it buys in interpretability it loses in resolution: the interference
/// fringes that make the Wigner function negative are precisely what the
/// smoothing erases.
///
/// # Errors
/// Returns an error for a non-positive spacing, width, or `hbar`.
pub fn husimi_q(
    psi: &[Complex],
    dx: f64,
    x0: f64,
    x: f64,
    p: f64,
    sigma: f64,
    hbar: f64,
) -> Result<f64, GeomError> {
    if psi.is_empty() || !(dx > 0.0) || !(sigma > 0.0) || !(hbar > 0.0) {
        return Err(GeomError::InvalidArgument("husimi_q: bad grid"));
    }
    // Q = |<coherent | psi>|^2 / (2 pi hbar), with the coherent state a
    // Gaussian of width sigma centred at (x, p).
    let normalisation = 1.0 / (2.0 * std::f64::consts::PI * sigma * sigma).powf(0.25);
    let mut acc = Complex::new(0.0, 0.0);
    for (k, z) in psi.iter().enumerate() {
        let xk = x0 + k as f64 * dx;
        let envelope =
            normalisation * (-(xk - x) * (xk - x) / (4.0 * sigma * sigma)).exp();
        let angle = -p * xk / hbar;
        let coherent = Complex::new(envelope * angle.cos(), envelope * angle.sin());
        acc = acc + coherent.conjugate() * *z;
    }
    let overlap = scale(acc, dx);
    Ok(overlap.norm_sq() / (2.0 * std::f64::consts::PI * hbar))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Midpoint quadrature over `[a, b]`, which is what the grid sums above
    /// approximate and what the closed forms below are integrated with.
    fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        (0..n).map(|k| f(a + (k as f64 + 0.5) * h)).sum::<f64>() * h
    }

    // -----------------------------------------------------------------
    // Orthogonal polynomials
    // -----------------------------------------------------------------

    #[test]
    fn the_hermite_recurrence_reproduces_the_closed_forms_and_the_roots() {
        // The first few are known outright, so the recurrence is checked
        // against arithmetic rather than against itself.
        for x in [-2.0f64, -0.5, 0.0, 0.3, 1.7, 4.0] {
            assert!(close(hermite_polynomial(0, x), 1.0, 1e-12));
            assert!(close(hermite_polynomial(1, x), 2.0 * x, 1e-12));
            assert!(close(hermite_polynomial(2, x), 4.0 * x * x - 2.0, 1e-12));
            assert!(close(hermite_polynomial(3, x), 8.0 * x * x * x - 12.0 * x, 1e-12));
            assert!(
                close(
                    hermite_polynomial(4, x),
                    16.0 * x.powi(4) - 48.0 * x * x + 12.0,
                    1e-11
                ),
                "H_4({x}) is {}",
                hermite_polynomial(4, x)
            );
        }
        // Parity: H_n(-x) = (-1)^n H_n(x), which the recurrence does not
        // impose and so could break.
        for n in 0..12 {
            for x in [0.4f64, 1.1, 2.6] {
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                assert!(
                    close(hermite_polynomial(n, -x), sign * hermite_polynomial(n, x), 1e-9),
                    "parity fails at n = {n}"
                );
            }
        }
        // H_n has exactly n real roots, all simple: counting sign changes on
        // a fine grid over the interval that contains them recovers n.
        for n in 1..=8 {
            let reach = 2.0 * (n as f64).sqrt() + 2.0;
            // Sampled at midpoints: an endpoint-anchored grid over a
            // symmetric interval lands exactly on x = 0, where the product of
            // neighbouring values is zero rather than negative and the root
            // at the origin goes uncounted.
            let steps = 20_000;
            let h = 2.0 * reach / steps as f64;
            let mut changes = 0;
            let mut previous = hermite_polynomial(n, -reach + 0.5 * h);
            for k in 1..steps {
                let x = -reach + (k as f64 + 0.5) * h;
                let value = hermite_polynomial(n, x);
                if previous * value < 0.0 {
                    changes += 1;
                }
                previous = value;
            }
            assert_eq!(changes, n, "H_{n} should have {n} roots");
        }
    }

    #[test]
    fn the_laguerre_recurrence_matches_its_closed_forms() {
        for x in [0.0f64, 0.5, 2.0, 6.0] {
            for k in [0.0f64, 1.0, 3.0] {
                assert!(close(laguerre_associated(0, k, x), 1.0, 1e-12));
                assert!(close(laguerre_associated(1, k, x), 1.0 + k - x, 1e-12));
                let l2 = x * x / 2.0 - (k + 2.0) * x + (k + 1.0) * (k + 2.0) / 2.0;
                assert!(
                    close(laguerre_associated(2, k, x), l2, 1e-11),
                    "L_2^{k}({x}) is {}, not {l2}",
                    laguerre_associated(2, k, x)
                );
            }
        }
        // The value at zero is the binomial coefficient C(n + k, n).
        for n in 0..8usize {
            for k in 0..4usize {
                let mut expected = 1.0;
                for j in 1..=n {
                    expected *= (n + k - j + 1) as f64 / j as f64;
                }
                assert!(
                    close(laguerre_associated(n, k as f64, 0.0), expected, 1e-9),
                    "L_{n}^{k}(0) is {}, not {expected}",
                    laguerre_associated(n, k as f64, 0.0)
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // Eigenstates
    // -----------------------------------------------------------------

    #[test]
    fn the_oscillator_eigenstates_are_orthonormal_and_solve_their_own_equation() {
        // Orthonormality is an integral identity that nothing in the
        // construction enforces -- the normalisation is put in by hand -- so
        // it is a real check on both the constant and the polynomial.
        let (mass, omega, hbar) = (1.0f64, 1.0f64, 1.0f64);
        for n in 0..6usize {
            for m in 0..6usize {
                let overlap = integrate(
                    |x| {
                        harmonic_oscillator_eigenstate(n, x, mass, omega, hbar)
                            * harmonic_oscillator_eigenstate(m, x, mass, omega, hbar)
                    },
                    -12.0,
                    12.0,
                    40_000,
                );
                let expected = f64::from(n == m);
                assert!(
                    close(overlap, expected, 1e-8),
                    "<{n}|{m}> is {overlap}, not {expected}"
                );
            }
        }

        // And each really is an eigenstate: applying H by finite difference
        // reproduces (n + 1/2) hbar omega times the state itself.
        let h = 1e-4;
        for n in 0..5usize {
            for &x in &[-1.3f64, 0.35, 2.1] {
                let psi = |y: f64| harmonic_oscillator_eigenstate(n, y, mass, omega, hbar);
                let second = (psi(x + h) - 2.0 * psi(x) + psi(x - h)) / (h * h);
                let applied = -hbar * hbar / (2.0 * mass) * second
                    + 0.5 * mass * omega * omega * x * x * psi(x);
                let expected = harmonic_oscillator_energy(n, omega, hbar) * psi(x);
                assert!(
                    close(applied, expected, 1e-5 * (1.0 + expected.abs())),
                    "n = {n} at x = {x}: H psi is {applied}, E psi is {expected}"
                );
            }
        }
        // The zero-point energy is not zero.
        assert!(close(harmonic_oscillator_energy(0, 3.0, 2.0), 3.0, 1e-12));
    }

    #[test]
    fn the_infinite_well_states_are_orthonormal_with_the_textbook_spectrum() {
        let l = 2.3f64;
        for n in 1..=6usize {
            for m in 1..=6usize {
                let overlap = integrate(
                    |x| infinite_well_eigenstate(n, x, l) * infinite_well_eigenstate(m, x, l),
                    0.0,
                    l,
                    50_000,
                );
                assert!(close(overlap, f64::from(n == m), 1e-6), "<{n}|{m}> is {overlap}");
            }
            // The energies grow as n^2, exactly.
            let ratio = infinite_well_energy(n, l, 1.0, 1.0) / infinite_well_energy(1, l, 1.0, 1.0);
            assert!(close(ratio, (n * n) as f64, 1e-12), "the ratio at n = {n} is {ratio}");
        }
        // Outside the well the state vanishes, and it vanishes at the walls.
        assert_eq!(infinite_well_eigenstate(1, -0.1, l), 0.0);
        assert_eq!(infinite_well_eigenstate(1, l + 0.1, l), 0.0);
        assert_eq!(infinite_well_eigenstate(1, 0.0, l), 0.0);
    }

    #[test]
    fn the_hydrogen_radial_states_are_normalised_and_have_the_right_node_count() {
        // The radial functions integrate to one against r^2 dr, and R_{n,l}
        // has exactly n - l - 1 nodes. Neither is imposed by the formula.
        let a0 = 1.0f64;
        for n in 1..=4usize {
            for l in 0..n {
                let total = integrate(
                    |r| {
                        let value = hydrogen_radial(n, l, r, a0);
                        value * value * r * r
                    },
                    0.0,
                    60.0 * n as f64,
                    400_000,
                );
                assert!(close(total, 1.0, 1e-5), "R_{n},{l} integrates to {total}");

                // Midpoints again: R_{4,2}'s only node sits at r = 12, which
                // an endpoint-anchored grid over [0, 240] hits exactly.
                let mut nodes = 0;
                let steps = 200_000;
                let reach = 60.0 * n as f64;
                let h = reach / steps as f64;
                let mut previous = hydrogen_radial(n, l, 0.5 * h, a0);
                for k in 1..steps {
                    let r = (k as f64 + 0.5) * h;
                    let value = hydrogen_radial(n, l, r, a0);
                    if previous * value < 0.0 {
                        nodes += 1;
                    }
                    previous = value;
                }
                assert_eq!(nodes, n - l - 1, "R_{n},{l} should have {} nodes", n - l - 1);
            }
        }

        // Orthogonality between states of the same l and different n.
        let overlap = integrate(
            |r| hydrogen_radial(1, 0, r, a0) * hydrogen_radial(2, 0, r, a0) * r * r,
            0.0,
            80.0,
            400_000,
        );
        assert!(close(overlap, 0.0, 1e-6), "<1s|2s> is {overlap}");

        // The energies follow -13.6 / n^2 and converge to zero.
        assert!(close(hydrogen_energy(1), -13.605_693_122_994, 1e-9));
        for n in 1..=6usize {
            assert!(close(hydrogen_energy(n) * (n * n) as f64, hydrogen_energy(1), 1e-9));
        }
    }

    #[test]
    fn the_hydrogen_orbitals_of_a_shell_sum_to_a_spherically_symmetric_density() {
        // Unsold's theorem: summing |Y_lm|^2 over m at fixed l gives a
        // constant. It is what makes a closed shell spherical, and it holds
        // for the real harmonics as well as the complex ones.
        let a0 = 1.0f64;
        for (n, l) in [(2usize, 1usize), (3, 1), (3, 2), (4, 2)] {
            for &(theta, phi) in
                &[(0.3f64, 0.9f64), (1.2, 2.6), (2.9, 0.1), (std::f64::consts::FRAC_PI_2, 4.4)]
            {
                let total: f64 = (-(l as i32)..=(l as i32))
                    .map(|m| hydrogen_orbital_density(n, l, m, 1.7, theta, phi, a0))
                    .sum();
                let radial = hydrogen_radial(n, l, 1.7, a0);
                let expected =
                    radial * radial * (2 * l + 1) as f64 / (4.0 * std::f64::consts::PI);
                assert!(
                    close(total, expected, 1e-9),
                    "the {n},{l} shell is not spherical: {total} against {expected}"
                );
            }
        }
        // Every orbital density is non-negative, which is the one thing a
        // density must be.
        for m in -1i32..=1 {
            assert!(hydrogen_orbital_density(2, 1, m, 2.0, 0.4, 1.1, a0) >= 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Wavefunctions on a grid
    // -----------------------------------------------------------------

    fn gaussian_grid(sigma: f64, k0: f64) -> Wavefunction1D {
        let n = 2048usize;
        let dx = 40.0 / n as f64;
        Wavefunction1D::gaussian_packet(0.0, k0, sigma, dx, -20.0, n).unwrap()
    }

    #[test]
    fn a_gaussian_packet_saturates_the_uncertainty_bound_and_nothing_else_does() {
        // The equality case is the whole content of the theorem's sharpness.
        for sigma in [0.4f64, 0.8, 1.5, 3.0] {
            let packet = gaussian_grid(sigma, 0.0);
            assert!(close(packet.norm(), 1.0, 1e-12), "the packet is not normalised");
            assert!(close(packet.variance_x().sqrt(), sigma, 1e-6), "the width is wrong");
            let product = packet.uncertainty_product(1.0).unwrap();
            assert!(
                close(product, 0.5, 1e-6),
                "sigma = {sigma}: the product is {product}, not hbar / 2"
            );
        }

        // A state built from two separated Gaussians has a strictly larger
        // product, as the theorem requires.
        let n = 2048usize;
        let dx = 40.0 / n as f64;
        let psi: Vec<Complex> = (0..n)
            .map(|k| {
                let x = -20.0 + k as f64 * dx;
                let left = (-(x + 4.0) * (x + 4.0) / 2.0).exp();
                let right = (-(x - 4.0) * (x - 4.0) / 2.0).exp();
                Complex::new(left + right, 0.0)
            })
            .collect();
        let mut cat = Wavefunction1D::new(psi, dx, -20.0).unwrap();
        cat.normalize();
        let product = cat.uncertainty_product(1.0).unwrap();
        assert!(product > 0.5, "the product is {product}, below the bound");
        // Two lumps at +/-2 of width 1/sqrt(2) give sigma_x = sqrt(4.5) and
        // sigma_k = 1/sqrt(2), for a product near 2.9 -- nearly six times the
        // bound.
        assert!(product > 2.5, "two separated peaks should be far from minimal: {product}");
    }

    #[test]
    fn the_packet_carries_the_momentum_it_was_given() {
        // A boost multiplies the packet by a phase, which cannot change the
        // position density but must shift the momentum by exactly k0.
        for k0 in [-3.0f64, -0.5, 0.0, 2.0, 5.5] {
            let packet = gaussian_grid(1.0, k0);
            let mean = packet.expectation_k().unwrap();
            assert!(close(mean, k0, 1e-6), "the mean wavenumber is {mean}, not {k0}");
            // The width in momentum is 1 / (2 sigma) whatever the boost.
            let spread = packet.variance_k().unwrap().sqrt();
            assert!(close(spread, 0.5, 1e-6), "the momentum width is {spread}");

            let still = gaussian_grid(1.0, 0.0);
            for (a, b) in packet.probability_density().iter().zip(still.probability_density()) {
                assert!((a - b).abs() < 1e-12, "the boost changed the position density");
            }
        }
    }

    #[test]
    fn a_free_packet_spreads_at_the_rate_the_closed_form_gives() {
        // sigma(t)^2 = sigma0^2 + (hbar t / 2 m sigma0)^2. The spreading is
        // not dissipation -- the evolution is unitary and reversible -- it is
        // the different momentum components separating.
        let sigma0 = 1.0f64;
        let packet = gaussian_grid(sigma0, 0.0);
        for t in [0.0f64, 0.5, 1.0, 2.0, 4.0] {
            let moved = packet.propagate_free(t, 1.0, 1.0).unwrap();
            assert!(close(moved.norm(), 1.0, 1e-12), "the norm changed to {}", moved.norm());
            let expected = (sigma0 * sigma0 + (t / (2.0 * sigma0)).powi(2)).sqrt();
            let width = moved.variance_x().sqrt();
            assert!(
                close(width, expected, 2e-4),
                "at t = {t} the width is {width}, not {expected}"
            );
        }

        // A boosted packet's centre moves at the group velocity hbar k / m.
        let packet = gaussian_grid(1.5, 2.0);
        let moved = packet.propagate_free(3.0, 1.0, 1.0).unwrap();
        assert!(
            close(moved.expectation_x(), 6.0, 2e-3),
            "the centre is at {}, not 6",
            moved.expectation_x()
        );
        // And the momentum distribution is untouched: the free Hamiltonian
        // commutes with itself.
        assert!(close(moved.expectation_k().unwrap(), 2.0, 1e-6));
        assert!(close(
            moved.variance_k().unwrap(),
            packet.variance_k().unwrap(),
            1e-12
        ));
    }

    #[test]
    fn the_energy_of_an_eigenstate_is_its_eigenvalue() {
        // Sampling a harmonic oscillator eigenstate on a grid and asking for
        // its energy must return (n + 1/2) hbar omega. Both the spectral
        // kinetic term and the potential term have to be right for that to
        // come out, and an error in either shows up immediately.
        let n_grid = 2048usize;
        let dx = 24.0 / n_grid as f64;
        let x0 = -12.0;
        let v: Vec<f64> = (0..n_grid).map(|k| 0.5 * (x0 + k as f64 * dx).powi(2)).collect();
        for n in 0..6usize {
            let psi: Vec<Complex> = (0..n_grid)
                .map(|k| {
                    Complex::new(
                        harmonic_oscillator_eigenstate(n, x0 + k as f64 * dx, 1.0, 1.0, 1.0),
                        0.0,
                    )
                })
                .collect();
            let state = Wavefunction1D::new(psi, dx, x0).unwrap();
            let energy = state.energy(&v, 1.0, 1.0).unwrap();
            let expected = n as f64 + 0.5;
            assert!(
                close(energy, expected, 1e-6),
                "state {n} has energy {energy}, not {expected}"
            );
        }

        // A free packet's energy is hbar^2 (k0^2 + 1 / 4 sigma^2) / 2m: the
        // motion plus the spread.
        let sigma = 1.2f64;
        let k0 = 1.7f64;
        let packet = gaussian_grid(sigma, k0);
        let free = vec![0.0; packet.len()];
        let expected = 0.5 * (k0 * k0 + 1.0 / (4.0 * sigma * sigma));
        let energy = packet.energy(&free, 1.0, 1.0).unwrap();
        assert!(close(energy, expected, 1e-6), "the packet's energy is {energy}, not {expected}");
    }

    #[test]
    fn overlaps_reproduce_orthonormality_on_the_grid() {
        let n_grid = 1024usize;
        let dx = 20.0 / n_grid as f64;
        let x0 = -10.0;
        let state = |n: usize| {
            let psi: Vec<Complex> = (0..n_grid)
                .map(|k| {
                    Complex::new(
                        harmonic_oscillator_eigenstate(n, x0 + k as f64 * dx, 1.0, 1.0, 1.0),
                        0.0,
                    )
                })
                .collect();
            Wavefunction1D::new(psi, dx, x0).unwrap()
        };
        for n in 0..5usize {
            for m in 0..5usize {
                let value = state(n).overlap(&state(m)).unwrap();
                assert!(close(value.re, f64::from(n == m), 1e-8), "<{n}|{m}> is {value:?}");
                assert!(close(value.im, 0.0, 1e-12));
            }
        }
        // A state overlaps itself by its own norm squared.
        let packet = gaussian_grid(1.0, 1.0);
        let self_overlap = packet.overlap(&packet).unwrap();
        assert!(close(self_overlap.re, 1.0, 1e-10) && close(self_overlap.im, 0.0, 1e-12));

        // And the overlap is conjugate-symmetric.
        let other = gaussian_grid(1.0, -1.0);
        let forward = packet.overlap(&other).unwrap();
        let backward = other.overlap(&packet).unwrap();
        assert!(close(forward.re, backward.re, 1e-12));
        assert!(close(forward.im, -backward.im, 1e-12));
    }

    // -----------------------------------------------------------------
    // Fock states
    // -----------------------------------------------------------------

    #[test]
    fn a_coherent_state_has_poisson_photon_statistics() {
        // Mean and variance both equal |alpha|^2, which is the signature that
        // separates a laser from a thermal source.
        for magnitude in [0.5f64, 1.0, 2.5, 4.0] {
            let coefficients = coherent_state(Complex::new(magnitude, 0.0), 120).unwrap();
            let weights: Vec<f64> = coefficients.iter().map(|z| z.norm_sq()).collect();
            let total: f64 = weights.iter().sum();
            assert!(close(total, 1.0, 1e-9), "the state has norm {total}");

            let mean: f64 = weights.iter().enumerate().map(|(n, w)| n as f64 * w).sum();
            let second: f64 =
                weights.iter().enumerate().map(|(n, w)| (n * n) as f64 * w).sum();
            let variance = second - mean * mean;
            let expected = magnitude * magnitude;
            assert!(close(mean, expected, 1e-6), "the mean is {mean}, not {expected}");
            assert!(
                close(variance, expected, 1e-6),
                "Poisson requires variance = mean: {variance} against {expected}"
            );

            // Each coefficient against the closed form.
            for (n, z) in coefficients.iter().enumerate().take(8) {
                let mut factorial = 1.0;
                for k in 1..=n {
                    factorial *= k as f64;
                }
                let predicted = (-expected / 2.0).exp() * magnitude.powi(n as i32)
                    / factorial.sqrt();
                assert!(close(z.re, predicted, 1e-9), "coefficient {n} is {}", z.re);
            }
        }
        // A phase on alpha becomes a phase on each coefficient and leaves the
        // statistics alone.
        let rotated = coherent_state(Complex::new(0.0, 2.0), 60).unwrap();
        let plain = coherent_state(Complex::new(2.0, 0.0), 60).unwrap();
        for (a, b) in rotated.iter().zip(&plain) {
            assert!(close(a.norm(), b.norm(), 1e-12));
        }
        assert!(coherent_state(Complex::new(1.0, 0.0), 0).is_err());
    }

    #[test]
    fn a_squeezed_vacuum_occupies_only_the_even_photon_numbers() {
        for r in [0.2f64, 0.5, 1.0] {
            let coefficients = squeezed_state(r, 0.4, 200).unwrap();
            for (n, z) in coefficients.iter().enumerate() {
                if n % 2 == 1 {
                    assert!(z.norm() < 1e-15, "the odd coefficient {n} is {}", z.norm());
                }
            }
            let total: f64 = coefficients.iter().map(|z| z.norm_sq()).sum();
            assert!(close(total, 1.0, 1e-6), "at r = {r} the norm is {total}");

            // The mean photon number of a squeezed vacuum is sinh^2 r: the
            // state contains photons despite being a vacuum in the squeezed
            // quadrature.
            let mean: f64 = coefficients
                .iter()
                .enumerate()
                .map(|(n, z)| n as f64 * z.norm_sq())
                .sum();
            assert!(
                close(mean, r.sinh() * r.sinh(), 1e-5),
                "at r = {r} the mean is {mean}, not {}",
                r.sinh() * r.sinh()
            );
        }
        // No squeezing leaves the vacuum.
        let none = squeezed_state(0.0, 0.0, 20).unwrap();
        assert!(close(none[0].norm(), 1.0, 1e-12));
        assert!(none[1..].iter().all(|z| z.norm() < 1e-15));
        assert!(squeezed_state(1.0, 0.0, 0).is_err());
    }

    // -----------------------------------------------------------------
    // Phase space
    // -----------------------------------------------------------------

    #[test]
    fn the_wigner_marginals_are_the_position_and_momentum_densities() {
        // The defining property, and the reason the Wigner function is worth
        // computing at all despite not being a probability density.
        let n = 256usize;
        let dx = 12.0 / n as f64;
        let x0 = -6.0;
        let packet = Wavefunction1D::gaussian_packet(0.0, 1.0, 1.0, dx, x0, n).unwrap();

        // Integrating over p at fixed x must give |psi(x)|^2.
        let p_max = 12.0f64;
        let p_steps = 800usize;
        let dp = 2.0 * p_max / p_steps as f64;
        for &index in &[100usize, 128, 150] {
            let x = x0 + index as f64 * dx;
            let marginal: f64 = (0..p_steps)
                .map(|j| {
                    let p = -p_max + (j as f64 + 0.5) * dp;
                    wigner_function(&packet.psi, dx, x0, x, p, 1.0).unwrap()
                })
                .sum::<f64>()
                * dp;
            let density = packet.psi[index].norm_sq();
            assert!(
                close(marginal, density, 1e-3 * (1.0 + density)),
                "at x = {x} the marginal is {marginal}, the density {density}"
            );
        }

        // A Gaussian's Wigner function is a Gaussian and never goes negative;
        // a superposition's does, and that is the point.
        for &(x, p) in &[(0.0f64, 1.0f64), (1.0, 0.5), (-2.0, 2.0)] {
            assert!(
                wigner_function(&packet.psi, dx, x0, x, p, 1.0).unwrap() > -1e-6,
                "a Gaussian's Wigner function went negative"
            );
        }
        let psi: Vec<Complex> = (0..n)
            .map(|k| {
                let x = x0 + k as f64 * dx;
                Complex::new(
                    (-(x + 2.0) * (x + 2.0) / 2.0).exp() + (-(x - 2.0) * (x - 2.0) / 2.0).exp(),
                    0.0,
                )
            })
            .collect();
        let mut cat = Wavefunction1D::new(psi, dx, x0).unwrap();
        cat.normalize();
        let lowest = (0..40)
            .map(|j| {
                let p = j as f64 * 0.1;
                wigner_function(&cat.psi, dx, x0, 0.0, p, 1.0).unwrap()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(lowest < -0.05, "a Schrodinger cat's Wigner function should go negative: {lowest}");
        assert!(wigner_function(&[], 1.0, 0.0, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn the_husimi_function_is_a_genuine_probability_density() {
        // Smoothing the Wigner function over a minimum-uncertainty cell
        // removes the negativity. That is what the Q function buys, and what
        // it costs is the interference structure the negativity encoded.
        let n = 256usize;
        let dx = 12.0 / n as f64;
        let x0 = -6.0;
        let psi: Vec<Complex> = (0..n)
            .map(|k| {
                let x = x0 + k as f64 * dx;
                Complex::new(
                    (-(x + 2.0) * (x + 2.0) / 2.0).exp() + (-(x - 2.0) * (x - 2.0) / 2.0).exp(),
                    0.0,
                )
            })
            .collect();
        let mut cat = Wavefunction1D::new(psi, dx, x0).unwrap();
        cat.normalize();

        let mut total = 0.0;
        let (dp, p_max) = (0.05f64, 6.0f64);
        for i in 0..n {
            let x = x0 + i as f64 * dx;
            let mut j = -p_max;
            while j < p_max {
                let q = husimi_q(&cat.psi, dx, x0, x, j + dp / 2.0, 0.5, 1.0).unwrap();
                assert!(q >= -1e-12, "the Q function went negative at ({x}, {j}): {q}");
                total += q * dx * dp;
                j += dp;
            }
        }
        assert!(close(total, 1.0, 1e-3), "the Q function integrates to {total}");

        // It peaks where the two lumps are, not between them -- though only
        // by a factor of about three and a half. The smoothing cell is not
        // small compared with the separation, and the resolution the Q
        // function gives up is real: the Wigner function above distinguishes
        // the same two lumps by an interference fringe that swings negative.
        let middle = husimi_q(&cat.psi, dx, x0, 0.0, 0.0, 0.5, 1.0).unwrap();
        let lump = husimi_q(&cat.psi, dx, x0, 2.0, 0.0, 0.5, 1.0).unwrap();
        assert!(lump > 3.0 * middle, "the Q function does not resolve the lumps");

        // Against a closed form, where one exists. Smoothing a Gaussian of
        // width s by a coherent state of width g gives, at zero momentum,
        // an overlap of C_g C_psi sqrt(pi / (a + b)) exp(-a b x^2 / (a + b))
        // with a = 1 / 4g^2 and b = 1 / 4s^2 -- another Gaussian, wider than
        // either. Nothing in the implementation knows that.
        let s = 0.8f64;
        let smooth = 0.6f64;
        let packet = Wavefunction1D::gaussian_packet(0.0, 0.0, s, dx, x0, n).unwrap();
        let a = 1.0 / (4.0 * smooth * smooth);
        let b = 1.0 / (4.0 * s * s);
        let c_g = (2.0 * std::f64::consts::PI * smooth * smooth).powf(-0.25);
        let c_psi = (2.0 * std::f64::consts::PI * s * s).powf(-0.25);
        for x in [-1.5f64, -0.5, 0.0, 0.7, 2.0] {
            let overlap = c_g
                * c_psi
                * (std::f64::consts::PI / (a + b)).sqrt()
                * (-a * b * x * x / (a + b)).exp();
            let expected = overlap * overlap / (2.0 * std::f64::consts::PI);
            let got = husimi_q(&packet.psi, dx, x0, x, 0.0, smooth, 1.0).unwrap();
            assert!(
                close(got, expected, 1e-6),
                "at x = {x} the Q function is {got}, the closed form {expected}"
            );
        }
        assert!(husimi_q(&cat.psi, dx, x0, 0.0, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn the_constructors_refuse_degenerate_input() {
        assert!(Wavefunction1D::new(vec![], 1.0, 0.0).is_err());
        assert!(Wavefunction1D::new(vec![Complex::new(1.0, 0.0)], 0.0, 0.0).is_err());
        assert!(Wavefunction1D::gaussian_packet(0.0, 0.0, 0.0, 0.1, -1.0, 16).is_err());
        assert!(Wavefunction1D::plane_wave(1.0, 0.1, 0.0, 0).is_err());

        // A non-power-of-two grid has no FFT here, and the spectral
        // quantities say so rather than returning something wrong.
        let odd = Wavefunction1D::plane_wave(1.0, 0.1, 0.0, 30).unwrap();
        assert!(odd.momentum_space().is_err());
        assert!(odd.expectation_k().is_err());
        assert!(odd.variance_k().is_err());
        assert!(odd.uncertainty_product(1.0).is_err());
        assert!(odd.propagate_free(0.1, 1.0, 1.0).is_err());

        let good = Wavefunction1D::plane_wave(1.0, 0.1, 0.0, 32).unwrap();
        assert!(good.energy(&[0.0; 4], 1.0, 1.0).is_err());
        assert!(good.energy(&[0.0; 32], 1.0, 0.0).is_err());
        assert!(good.propagate_free(0.1, 1.0, -1.0).is_err());
        assert!(good.overlap(&odd).is_err());
        assert!(!good.is_empty() && good.len() == 32);

        // A zero wavefunction has no expectations to report and says so by
        // returning zeros rather than dividing by zero.
        let empty = Wavefunction1D::new(vec![Complex::new(0.0, 0.0); 8], 0.1, 0.0).unwrap();
        assert_eq!(empty.norm(), 0.0);
        assert_eq!(empty.expectation_x(), 0.0);
        assert_eq!(empty.variance_x(), 0.0);
        assert_eq!(empty.expectation_k().unwrap(), 0.0);
        assert_eq!(empty.variance_k().unwrap(), 0.0);
        assert_eq!(empty.energy(&[0.0; 8], 1.0, 1.0).unwrap(), 0.0);
        let mut still_empty = empty.clone();
        still_empty.normalize();
        assert!(still_empty.psi.iter().all(|z| z.norm() == 0.0));
    }

    #[test]
    #[should_panic(expected = "l < n")]
    fn hydrogen_rejects_an_impossible_angular_momentum() {
        let _ = hydrogen_radial(2, 2, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "indexed from one")]
    fn the_well_rejects_a_zeroth_state() {
        let _ = infinite_well_eigenstate(0, 0.5, 1.0);
    }
}
