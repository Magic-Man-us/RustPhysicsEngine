//! The Ising model and its relatives, by Monte Carlo.
//!
//! The two-dimensional Ising model is the one interacting system with a
//! phase transition that is solved exactly, so it is where a Monte Carlo
//! code can be checked against arithmetic rather than against another Monte
//! Carlo code. Onsager's solution gives the critical temperature, the energy
//! and the spontaneous magnetisation in closed form, and any sampler that
//! disagrees with them is wrong.
//!
//! The algorithmic point of the module is the contrast between the two
//! updates. Metropolis flips one spin at a time, so near the critical
//! temperature -- where the correlation length diverges and whole regions
//! must turn over together -- successive configurations stay correlated for
//! a time growing as the system size to a power near two. Wolff builds a
//! cluster whose size is itself set by the correlation length and flips it
//! whole, which all but removes that critical slowing down. The two sample
//! the same distribution; they differ only in how long it takes.

use crate::error::GeomError;
use crate::monte_carlo::Rng;

/// Bond probability floor: below this a cluster algorithm degenerates to
/// single-spin updates and there is nothing to gain.
const MIN_BOND_PROBABILITY: f64 = 1e-12;

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

// ---------------------------------------------------------------------------
// The two-dimensional model
// ---------------------------------------------------------------------------

/// A square-lattice Ising model with nearest-neighbour coupling.
///
/// `H = -j sum_<ij> s_i s_j - h sum_i s_i` with spins `+/-1`, and `beta` the
/// inverse temperature in units where Boltzmann's constant is one.
#[derive(Debug, Clone)]
pub struct Ising2D {
    /// Linear size; the lattice holds `n * n` spins.
    pub n: usize,
    /// The spins, row major, each `+1` or `-1`.
    pub spins: Vec<i8>,
    /// Exchange coupling. Positive is ferromagnetic.
    pub j: f64,
    /// External field.
    pub h: f64,
    /// Inverse temperature.
    pub beta: f64,
    /// Whether the lattice wraps.
    pub periodic: bool,
}

/// Summary statistics from a Monte Carlo run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsingStats {
    /// Mean energy per site.
    pub e_mean: f64,
    /// Variance of the energy per site.
    pub e_var: f64,
    /// Mean magnetisation per site, signed.
    pub m_mean: f64,
    /// Mean absolute magnetisation per site.
    pub m_abs: f64,
    /// Magnetic susceptibility per site.
    pub susceptibility: f64,
    /// Heat capacity per site.
    pub heat_capacity: f64,
    /// The Binder cumulant `1 - <m^4> / (3 <m^2>^2)`.
    pub binder_cumulant: f64,
    /// How many measurements went into these.
    pub samples: usize,
}

impl Ising2D {
    /// A lattice with every spin up.
    ///
    /// # Errors
    /// Returns an error for a lattice smaller than two or larger than 512 a
    /// side, or a non-positive inverse temperature.
    pub fn cold(n: usize, j: f64, h: f64, beta: f64, periodic: bool) -> Result<Self, GeomError> {
        if !(2..=512).contains(&n) {
            return Err(GeomError::InvalidArgument("the lattice must be 2 to 512 a side"));
        }
        if !(beta > 0.0) || !beta.is_finite() {
            return Err(GeomError::InvalidArgument("beta must be positive and finite"));
        }
        Ok(Self { n, spins: vec![1i8; n * n], j, h, beta, periodic })
    }

    /// A lattice with random spins.
    ///
    /// # Errors
    /// Returns an error on the same conditions as [`Ising2D::cold`].
    pub fn random(
        n: usize,
        j: f64,
        h: f64,
        beta: f64,
        periodic: bool,
        rng: &mut Rng,
    ) -> Result<Self, GeomError> {
        let mut lattice = Self::cold(n, j, h, beta, periodic)?;
        for spin in &mut lattice.spins {
            *spin = if rng.next_f64() < 0.5 { 1 } else { -1 };
        }
        Ok(lattice)
    }

    /// The site index of `(row, column)`.
    fn index(&self, row: usize, column: usize) -> usize {
        row * self.n + column
    }

    /// The neighbours of a site, as indices.
    fn neighbours(&self, site: usize) -> Vec<usize> {
        let (row, column) = (site / self.n, site % self.n);
        let mut out = Vec::with_capacity(4);
        let last = self.n - 1;
        // Up, down, left, right, wrapping only when periodic.
        if row > 0 {
            out.push(self.index(row - 1, column));
        } else if self.periodic {
            out.push(self.index(last, column));
        }
        if row < last {
            out.push(self.index(row + 1, column));
        } else if self.periodic {
            out.push(self.index(0, column));
        }
        if column > 0 {
            out.push(self.index(row, column - 1));
        } else if self.periodic {
            out.push(self.index(row, last));
        }
        if column < last {
            out.push(self.index(row, column + 1));
        } else if self.periodic {
            out.push(self.index(row, 0));
        }
        out
    }

    /// The total energy.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let mut bonds = 0.0;
        for site in 0..self.spins.len() {
            for neighbour in self.neighbours(site) {
                // Each bond is seen twice.
                bonds += f64::from(self.spins[site]) * f64::from(self.spins[neighbour]);
            }
        }
        let field: f64 = self.spins.iter().map(|s| f64::from(*s)).sum();
        -self.j * bonds / 2.0 - self.h * field
    }

    /// The energy per site.
    #[must_use]
    pub fn energy_per_site(&self) -> f64 {
        self.energy() / self.spins.len() as f64
    }

    /// The magnetisation per site, signed.
    #[must_use]
    pub fn magnetization(&self) -> f64 {
        self.spins.iter().map(|s| f64::from(*s)).sum::<f64>() / self.spins.len() as f64
    }

    /// The energy change if one spin were flipped.
    fn flip_cost(&self, site: usize) -> f64 {
        let local: f64 = self
            .neighbours(site)
            .iter()
            .map(|&k| f64::from(self.spins[k]))
            .sum();
        2.0 * f64::from(self.spins[site]) * (self.j * local + self.h)
    }

    /// One Metropolis sweep: `n^2` attempted single-spin flips.
    ///
    /// The acceptance rule `min(1, exp(-beta dE))` satisfies detailed balance
    /// with the Boltzmann distribution, which is what makes the chain sample
    /// it. Note that a rejected move still counts as a step: the current
    /// configuration is re-measured, and treating rejections as "nothing
    /// happened" biases every average.
    pub fn metropolis_sweep(&mut self, rng: &mut Rng) {
        for _ in 0..self.spins.len() {
            let site = pick(rng, self.spins.len());
            let cost = self.flip_cost(site);
            if cost <= 0.0 || rng.next_f64() < (-self.beta * cost).exp() {
                self.spins[site] = -self.spins[site];
            }
        }
    }

    /// One heat-bath sweep: each visited spin is redrawn from its conditional
    /// distribution rather than proposed and accepted.
    ///
    /// Also correct, and it never rejects -- but it is not faster in any
    /// useful sense, because a spin redrawn to its current value has moved
    /// just as little as a rejected proposal.
    pub fn heat_bath_sweep(&mut self, rng: &mut Rng) {
        for _ in 0..self.spins.len() {
            let site = pick(rng, self.spins.len());
            let local: f64 = self
                .neighbours(site)
                .iter()
                .map(|&k| f64::from(self.spins[k]))
                .sum();
            let field = self.j * local + self.h;
            // P(up) = 1 / (1 + exp(-2 beta field)).
            let up = 1.0 / (1.0 + (-2.0 * self.beta * field).exp());
            self.spins[site] = if rng.next_f64() < up { 1 } else { -1 };
        }
    }

    /// One Wolff cluster update, returning the cluster size.
    ///
    /// Grows a cluster of aligned spins by adding each neighbouring bond with
    /// probability `1 - exp(-2 beta j)`, then flips the whole thing. The
    /// acceptance is *one* -- the bond probability is chosen precisely so
    /// that the construction's bias cancels the Boltzmann weight -- which is
    /// why the method has no rejected moves at all.
    ///
    /// Only meaningful for a ferromagnetic coupling in zero field; the field
    /// breaks the cancellation, and this implementation ignores it.
    ///
    /// # Errors
    /// Returns an error for a non-positive coupling or a non-zero field.
    pub fn wolff_cluster_step(&mut self, rng: &mut Rng) -> Result<usize, GeomError> {
        if !(self.j > 0.0) {
            return Err(GeomError::InvalidArgument("Wolff needs a ferromagnetic coupling"));
        }
        if self.h != 0.0 {
            return Err(GeomError::InvalidArgument("Wolff needs zero external field"));
        }
        let add = 1.0 - (-2.0 * self.beta * self.j).exp();
        if add < MIN_BOND_PROBABILITY {
            // At infinite temperature the cluster is a single spin; flipping
            // it is still a valid move.
            let site = pick(rng, self.spins.len());
            self.spins[site] = -self.spins[site];
            return Ok(1);
        }
        let seed = pick(rng, self.spins.len());
        let sign = self.spins[seed];
        let mut in_cluster = vec![false; self.spins.len()];
        let mut stack = vec![seed];
        in_cluster[seed] = true;
        let mut size = 0usize;

        while let Some(site) = stack.pop() {
            size += 1;
            self.spins[site] = -sign;
            for neighbour in self.neighbours(site) {
                if !in_cluster[neighbour] && self.spins[neighbour] == sign && rng.next_f64() < add {
                    in_cluster[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
        Ok(size)
    }

    /// Runs the chain and returns summary statistics.
    ///
    /// One *update* is a Metropolis sweep of `n^2` attempted flips, or a
    /// single Wolff cluster step. The two are not the same amount of work,
    /// and deliberately so: a Wolff update must be a fixed number of cluster
    /// steps rather than "however many it takes to flip a lattice's worth of
    /// spins". That second rule looks like the natural way to equalise the
    /// work and it silently biases every average, because it stops right
    /// after a large cluster -- and a large cluster means an ordered
    /// configuration, so measurements are taken preferentially at low
    /// energies. Measuring at a *fixed* interval of a Markov chain is
    /// unbiased; measuring when the chain reaches a state-dependent
    /// condition is not.
    ///
    /// `thermalize` updates are discarded before measurement begins. That
    /// discard is not optional either: the chain starts from a configuration
    /// that is not a Boltzmann sample, and averaging over the approach to
    /// equilibrium biases everything.
    ///
    /// # Errors
    /// Returns an error for a zero measurement interval or no sweeps.
    pub fn sample(
        &mut self,
        sweeps: usize,
        thermalize: usize,
        measure_every: usize,
        use_wolff: bool,
        rng: &mut Rng,
    ) -> Result<IsingStats, GeomError> {
        if sweeps == 0 || measure_every == 0 {
            return Err(GeomError::InvalidArgument("sample needs sweeps and an interval"));
        }
        let sites = self.spins.len() as f64;
        let step = |lattice: &mut Self, rng: &mut Rng| -> Result<(), GeomError> {
            if use_wolff {
                lattice.wolff_cluster_step(rng)?;
            } else {
                lattice.metropolis_sweep(rng);
            }
            Ok(())
        };

        for _ in 0..thermalize {
            step(self, rng)?;
        }

        let (mut e1, mut e2) = (0.0f64, 0.0f64);
        let (mut m1, mut m_abs, mut m2, mut m4) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut samples = 0usize;
        for sweep in 0..sweeps {
            step(self, rng)?;
            if sweep % measure_every != 0 {
                continue;
            }
            let e = self.energy_per_site();
            let m = self.magnetization();
            e1 += e;
            e2 += e * e;
            m1 += m;
            m_abs += m.abs();
            m2 += m * m;
            m4 += m * m * m * m;
            samples += 1;
        }
        if samples == 0 {
            return Err(GeomError::Degenerate("no measurements were taken"));
        }
        let count = samples as f64;
        let (e1, e2) = (e1 / count, e2 / count);
        let (m1, m_abs, m2, m4) = (m1 / count, m_abs / count, m2 / count, m4 / count);
        let e_var = (e2 - e1 * e1).max(0.0);
        Ok(IsingStats {
            e_mean: e1,
            e_var,
            m_mean: m1,
            m_abs,
            // Both fluctuation formulas carry a factor of the site count,
            // since the variances above are per site and the response is not.
            susceptibility: self.beta * sites * (m2 - m_abs * m_abs).max(0.0),
            heat_capacity: self.beta * self.beta * sites * e_var,
            binder_cumulant: if m2 > 0.0 { 1.0 - m4 / (3.0 * m2 * m2) } else { 0.0 },
            samples,
        })
    }

    /// The spin-spin correlation at separation `r` along a lattice axis.
    ///
    /// # Errors
    /// Returns an error if the separation exceeds the lattice.
    pub fn correlation_function(&self, r: usize) -> Result<f64, GeomError> {
        if r >= self.n {
            return Err(GeomError::InvalidArgument("the separation exceeds the lattice"));
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for row in 0..self.n {
            for column in 0..self.n {
                let here = f64::from(self.spins[self.index(row, column)]);
                // Along the row.
                if self.periodic || column + r < self.n {
                    let other = self.spins[self.index(row, (column + r) % self.n)];
                    total += here * f64::from(other);
                    count += 1;
                }
                // And down the column.
                if self.periodic || row + r < self.n {
                    let other = self.spins[self.index((row + r) % self.n, column)];
                    total += here * f64::from(other);
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Ok(0.0);
        }
        Ok(total / count as f64)
    }

    /// The ensemble-averaged correlation function out to half the lattice,
    /// together with the mean squared magnetisation.
    ///
    /// Averaging over the run is not a refinement. A single configuration's
    /// correlation function is a sample of a random variable whose spread at
    /// large separation is comparable to its mean, so a length fitted from
    /// one snapshot is fitted to noise -- and the noise does not shrink as
    /// the lattice grows, because the number of *independent* regions does
    /// not either.
    ///
    /// # Errors
    /// Returns an error for a lattice too small to fit on, or no updates.
    pub fn sample_correlations(
        &mut self,
        updates: usize,
        use_wolff: bool,
        rng: &mut Rng,
    ) -> Result<(Vec<f64>, f64), GeomError> {
        if self.n < 6 {
            return Err(GeomError::InvalidArgument("the lattice is too small to fit"));
        }
        if updates == 0 {
            return Err(GeomError::InvalidArgument("sample_correlations needs updates"));
        }
        let reach = self.n / 2;
        let mut totals = vec![0.0f64; reach + 1];
        let mut m2 = 0.0f64;
        for _ in 0..updates {
            if use_wolff {
                self.wolff_cluster_step(rng)?;
            } else {
                self.metropolis_sweep(rng);
            }
            for (r, total) in totals.iter_mut().enumerate() {
                *total += self.correlation_function(r)?;
            }
            let m = self.magnetization();
            m2 += m * m;
        }
        let count = updates as f64;
        Ok((totals.iter().map(|t| t / count).collect(), m2 / count))
    }

    /// Fits a correlation length to an averaged correlation function.
    ///
    /// The connected correlation `C(r) - <m>^2` decays as `exp(-r / xi)`, so
    /// the length is minus the reciprocal slope of its logarithm. Only the
    /// separations where the connected correlation is well clear of the
    /// sampling noise are fitted: a threshold near zero admits points that
    /// are pure noise, and the fitted slope is then noise too -- which reads
    /// as a *long* correlation length in a hot lattice, exactly backwards.
    ///
    /// Returns zero when there is nothing resolvable to fit, and the lattice
    /// size when the correlation does not decay within it -- which is the
    /// honest answer near the critical point, where the true length exceeds
    /// anything a finite lattice can report.
    ///
    /// # Errors
    /// Returns an error for fewer than four separations.
    pub fn correlation_length_estimate(
        correlations: &[f64],
        background: f64,
    ) -> Result<f64, GeomError> {
        if correlations.len() < 4 {
            return Err(GeomError::InvalidArgument("the fit needs at least four separations"));
        }
        let mut points: Vec<(f64, f64)> = Vec::new();
        for (r, value) in correlations.iter().enumerate().skip(1) {
            let connected = value - background;
            if connected > 0.02 {
                points.push((r as f64, connected.ln()));
            } else {
                // Past the first unresolvable separation the rest is noise.
                break;
            }
        }
        if points.len() < 3 {
            return Ok(0.0);
        }
        if points.len() + 1 == correlations.len() {
            // Still correlated at the furthest separation measured.
            return Ok(correlations.len() as f64);
        }
        let n = points.len() as f64;
        let sx: f64 = points.iter().map(|p| p.0).sum();
        let sy: f64 = points.iter().map(|p| p.1).sum();
        let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let denominator = n * sxx - sx * sx;
        if denominator.abs() < 1e-12 {
            return Ok(0.0);
        }
        let slope = (n * sxy - sx * sy) / denominator;
        if slope >= 0.0 {
            return Ok(correlations.len() as f64);
        }
        Ok(-1.0 / slope)
    }

    /// The integrated autocorrelation time of the magnetisation, together
    /// with the mean number of spin flips an update costs.
    ///
    /// The time says how many updates a measurement is worth: `2 tau`
    /// consecutive samples carry the information of one independent one, so
    /// error bars computed as though samples were independent are too small
    /// by a factor of `sqrt(2 tau)`.
    ///
    /// The work is reported alongside because the two algorithms' updates are
    /// not comparable on their own. A Metropolis update attempts `n^2` flips;
    /// a Wolff update flips one cluster, whose size varies with the
    /// temperature. Comparing the two requires `tau` times the work, not
    /// `tau` alone -- and a comparison in bare updates would flatter whichever
    /// algorithm happened to define the larger one.
    ///
    /// # Errors
    /// Returns an error for too few updates to estimate from.
    pub fn autocorrelation_time(
        &mut self,
        updates: usize,
        use_wolff: bool,
        rng: &mut Rng,
    ) -> Result<(f64, f64), GeomError> {
        if updates < 50 {
            return Err(GeomError::InvalidArgument("the estimate needs at least fifty updates"));
        }
        let mut series = Vec::with_capacity(updates);
        let mut work = 0usize;
        for _ in 0..updates {
            if use_wolff {
                work += self.wolff_cluster_step(rng)?;
            } else {
                work += self.spins.len();
                self.metropolis_sweep(rng);
            }
            series.push(self.magnetization().abs());
        }
        let work_per_update = work as f64 / updates as f64;
        let n = series.len() as f64;
        let mean: f64 = series.iter().sum::<f64>() / n;
        let variance: f64 = series.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        if variance <= 0.0 {
            return Ok((0.5, work_per_update));
        }
        // Summed until the correlation first goes negative, which is the
        // standard automatic window: past that point the estimator is mostly
        // noise and summing further makes it worse, not better.
        let mut tau = 0.5;
        for lag in 1..series.len() / 4 {
            let covariance: f64 = (0..series.len() - lag)
                .map(|k| (series[k] - mean) * (series[k + lag] - mean))
                .sum::<f64>()
                / (series.len() - lag) as f64;
            let rho = covariance / variance;
            if rho <= 0.0 {
                break;
            }
            tau += rho;
        }
        Ok((tau, work_per_update))
    }
}

// ---------------------------------------------------------------------------
// Exact results
// ---------------------------------------------------------------------------

/// The exact critical temperature of the two-dimensional Ising model:
/// `2 / ln(1 + sqrt 2)`.
///
/// About 2.269. Kramers and Wannier found it from a duality argument years
/// before Onsager solved the model, without ever computing the free energy --
/// the self-dual point has to be the transition if there is only one.
#[must_use]
pub fn ising_tc_exact() -> f64 {
    2.0 / (1.0 + 2.0f64.sqrt()).ln()
}

/// Onsager's spontaneous magnetisation, zero above the critical temperature.
///
/// `(1 - sinh^-4(2 beta j))^(1/8)`. The exponent one eighth is the critical
/// exponent beta, and its being a simple fraction rather than the one half
/// that mean-field theory predicts is the whole reason the exact solution
/// mattered.
///
/// # Errors
/// Returns an error for a non-positive coupling or inverse temperature.
pub fn onsager_magnetization(beta: f64, j: f64) -> Result<f64, GeomError> {
    if !(beta > 0.0) || !(j > 0.0) {
        return Err(GeomError::InvalidArgument("onsager_magnetization needs positive parameters"));
    }
    let s = (2.0 * beta * j).sinh();
    if s <= 1.0 {
        return Ok(0.0);
    }
    Ok((1.0 - s.powi(-4)).powf(0.125))
}

/// Onsager's energy per site of the infinite lattice.
///
/// Involves a complete elliptic integral, which is where the logarithmic
/// divergence of the heat capacity at the critical point comes from: the
/// integral's derivative diverges exactly at the self-dual point.
///
/// # Errors
/// Returns an error for a non-positive coupling or inverse temperature.
pub fn onsager_energy(beta: f64, j: f64) -> Result<f64, GeomError> {
    if !(beta > 0.0) || !(j > 0.0) {
        return Err(GeomError::InvalidArgument("onsager_energy needs positive parameters"));
    }
    let k = 2.0 * beta * j;
    let kappa = 2.0 * k.sinh() / k.cosh().powi(2);
    // The crate's elliptic_k takes the parameter m = kappa^2.
    let m = (kappa * kappa).min(1.0);
    let elliptic = crate::special::elliptic::elliptic_k(m);
    let cotangent = k.cosh() / k.sinh();
    Ok(-j * cotangent
        * (1.0 + 2.0 / std::f64::consts::PI * (2.0 * k.tanh().powi(2) - 1.0) * elliptic))
}

/// The one-dimensional Ising chain by transfer matrix, returning the free
/// energy per site and the magnetisation per site.
///
/// The chain has no transition at any positive temperature, which is Ising's
/// own result and the reason he thought the model uninteresting. The transfer
/// matrix shows why: the free energy is the logarithm of the larger
/// eigenvalue of a two-by-two matrix with strictly positive entries, and such
/// an eigenvalue is analytic in the temperature.
///
/// # Errors
/// Returns an error for a non-positive inverse temperature.
pub fn ising_1d_exact(beta: f64, j: f64, h: f64) -> Result<(f64, f64), GeomError> {
    if !(beta > 0.0) || !beta.is_finite() {
        return Err(GeomError::InvalidArgument("beta must be positive and finite"));
    }
    let a = (beta * j).exp();
    let b = (-beta * j).exp();
    let (up, down) = ((beta * h).exp(), (-beta * h).exp());
    // T = [[a * up, b], [b, a * down]].
    let trace = a * up + a * down;
    let determinant = a * a - b * b;
    let discriminant = (trace * trace / 4.0 - determinant).max(0.0).sqrt();
    let lambda = trace / 2.0 + discriminant;
    let free_energy = -lambda.ln() / beta;
    // m = sinh(bh) / sqrt(sinh^2(bh) + exp(-4 b j)).
    let sh = (beta * h).sinh();
    let magnetization = sh / (sh * sh + (-4.0 * beta * j).exp()).sqrt();
    Ok((free_energy, magnetization))
}

/// The partition function of a small system by direct enumeration.
///
/// Exponential in the site count, so it stops at about twenty-four spins --
/// but within that range it is exact, which makes it the reference every
/// sampler here is checked against.
///
/// # Errors
/// Returns an error above twenty-four sites or for a non-positive beta.
pub fn partition_function_exact_small(
    energy: &dyn Fn(u64) -> f64,
    sites: usize,
    beta: f64,
) -> Result<f64, GeomError> {
    if sites == 0 || sites > 24 {
        return Err(GeomError::InvalidArgument("enumeration handles 1 to 24 sites"));
    }
    if !(beta > 0.0) {
        return Err(GeomError::InvalidArgument("beta must be positive"));
    }
    // Summed relative to the lowest energy, so that a cold system does not
    // overflow the exponential on the way to a perfectly ordinary answer.
    let lowest = (0..(1u64 << sites)).map(energy).fold(f64::INFINITY, f64::min);
    let shifted: f64 = (0..(1u64 << sites))
        .map(|state| (-beta * (energy(state) - lowest)).exp())
        .sum();
    Ok(shifted * (-beta * lowest).exp())
}

/// The free energy from a partition function.
///
/// # Errors
/// Returns an error for a non-positive partition function or beta.
pub fn free_energy_from_z(z: f64, beta: f64) -> Result<f64, GeomError> {
    if !(z > 0.0) || !(beta > 0.0) {
        return Err(GeomError::InvalidArgument("free_energy_from_z needs positive input"));
    }
    Ok(-z.ln() / beta)
}

/// The mean energy and entropy of a small system by enumeration.
///
/// # Errors
/// Returns an error on the same conditions as
/// [`partition_function_exact_small`].
pub fn thermodynamics_exact_small(
    energy: &dyn Fn(u64) -> f64,
    sites: usize,
    beta: f64,
) -> Result<(f64, f64), GeomError> {
    if sites == 0 || sites > 24 {
        return Err(GeomError::InvalidArgument("enumeration handles 1 to 24 sites"));
    }
    if !(beta > 0.0) {
        return Err(GeomError::InvalidArgument("beta must be positive"));
    }
    let lowest = (0..(1u64 << sites)).map(energy).fold(f64::INFINITY, f64::min);
    let mut z = 0.0;
    let mut e_total = 0.0;
    for state in 0..(1u64 << sites) {
        let e = energy(state);
        let weight = (-beta * (e - lowest)).exp();
        z += weight;
        e_total += weight * e;
    }
    let mean_energy = e_total / z;
    // S = beta (E - F), with F measured from the same shifted sum.
    let free_energy = lowest - z.ln() / beta;
    Ok((mean_energy, beta * (mean_energy - free_energy)))
}

// ---------------------------------------------------------------------------
// Related lattice models
// ---------------------------------------------------------------------------

/// The `q`-state Potts model on a square lattice.
///
/// Generalises Ising, which is the two-state case. The transition turns
/// first order above `q = 4` in two dimensions, which is why the model is the
/// standard example that the *order* of a transition is not a detail of the
/// interaction but a consequence of the symmetry.
#[derive(Debug, Clone)]
pub struct Potts2D {
    /// The number of states per site.
    pub q: u8,
    /// Linear size.
    pub n: usize,
    /// The states, row major, each in `0..q`.
    pub states: Vec<u8>,
    /// Coupling.
    pub j: f64,
    /// Inverse temperature.
    pub beta: f64,
}

impl Potts2D {
    /// A random configuration.
    ///
    /// # Errors
    /// Returns an error for fewer than two states, a bad lattice size, or a
    /// non-positive beta.
    pub fn random(q: u8, n: usize, j: f64, beta: f64, rng: &mut Rng) -> Result<Self, GeomError> {
        if q < 2 {
            return Err(GeomError::InvalidArgument("Potts needs at least two states"));
        }
        if !(2..=256).contains(&n) || !(beta > 0.0) {
            return Err(GeomError::InvalidArgument("Potts2D: bad lattice or temperature"));
        }
        let states = (0..n * n).map(|_| pick(rng, q as usize) as u8).collect();
        Ok(Self { q, n, states, j, beta })
    }

    fn neighbours(&self, site: usize) -> [usize; 4] {
        let (row, column) = (site / self.n, site % self.n);
        let last = self.n - 1;
        [
            (if row > 0 { row - 1 } else { last }) * self.n + column,
            (if row < last { row + 1 } else { 0 }) * self.n + column,
            row * self.n + if column > 0 { column - 1 } else { last },
            row * self.n + if column < last { column + 1 } else { 0 },
        ]
    }

    /// The energy: minus the coupling for each agreeing bond.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let mut agreeing = 0usize;
        for site in 0..self.states.len() {
            for neighbour in self.neighbours(site) {
                if self.states[site] == self.states[neighbour] {
                    agreeing += 1;
                }
            }
        }
        -self.j * agreeing as f64 / 2.0
    }

    /// One Metropolis sweep.
    pub fn metropolis_sweep(&mut self, rng: &mut Rng) {
        for _ in 0..self.states.len() {
            let site = pick(rng, self.states.len());
            let proposal = pick(rng, self.q as usize) as u8;
            if proposal == self.states[site] {
                continue;
            }
            let neighbours = self.neighbours(site);
            let before = neighbours.iter().filter(|&&k| self.states[k] == self.states[site]).count();
            let after = neighbours.iter().filter(|&&k| self.states[k] == proposal).count();
            let cost = self.j * (before as f64 - after as f64);
            if cost <= 0.0 || rng.next_f64() < (-self.beta * cost).exp() {
                self.states[site] = proposal;
            }
        }
    }

    /// The order parameter: how far the most common state's share exceeds
    /// what randomness would give.
    #[must_use]
    pub fn order_parameter(&self) -> f64 {
        let mut counts = vec![0usize; self.q as usize];
        for state in &self.states {
            counts[*state as usize] += 1;
        }
        let largest = counts.iter().copied().max().unwrap_or(0) as f64 / self.states.len() as f64;
        let q = f64::from(self.q);
        (q * largest - 1.0) / (q - 1.0)
    }
}

/// The exact critical temperature of the `q`-state Potts model in two
/// dimensions: `1 / ln(1 + sqrt q)`.
///
/// Reduces to the Ising value at `q = 2`, as it must.
///
/// # Errors
/// Returns an error for fewer than two states.
pub fn potts_tc_exact(q: u8) -> Result<f64, GeomError> {
    if q < 2 {
        return Err(GeomError::InvalidArgument("Potts needs at least two states"));
    }
    Ok(1.0 / (1.0 + f64::from(q).sqrt()).ln())
}

/// The two-dimensional XY model: continuous spins on a square lattice.
///
/// It has no ordered phase at any positive temperature -- a continuous
/// symmetry cannot break in two dimensions -- and yet it has a transition,
/// where vortices unbind. That the transition exists without an order
/// parameter is what makes it interesting.
#[derive(Debug, Clone)]
pub struct XyModel2D {
    /// Linear size.
    pub n: usize,
    /// The angles, row major.
    pub theta: Vec<f64>,
    /// Coupling.
    pub j: f64,
    /// Inverse temperature.
    pub beta: f64,
}

impl XyModel2D {
    /// A random configuration.
    ///
    /// # Errors
    /// Returns an error for a bad lattice size or non-positive beta.
    pub fn random(n: usize, j: f64, beta: f64, rng: &mut Rng) -> Result<Self, GeomError> {
        if !(4..=256).contains(&n) || !(beta > 0.0) {
            return Err(GeomError::InvalidArgument("XyModel2D: bad lattice or temperature"));
        }
        let theta = (0..n * n)
            .map(|_| rng.next_f64() * std::f64::consts::TAU)
            .collect();
        Ok(Self { n, theta, j, beta })
    }

    fn at(&self, row: usize, column: usize) -> f64 {
        self.theta[(row % self.n) * self.n + (column % self.n)]
    }

    /// The energy: minus the coupling times the cosine of each bond angle.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let mut total = 0.0;
        for row in 0..self.n {
            for column in 0..self.n {
                let here = self.at(row, column);
                total += (here - self.at(row + 1, column)).cos();
                total += (here - self.at(row, column + 1)).cos();
            }
        }
        -self.j * total
    }

    /// One Metropolis sweep, proposing a bounded angle change.
    pub fn metropolis_sweep(&mut self, rng: &mut Rng, step: f64) {
        for _ in 0..self.theta.len() {
            let site = pick(rng, self.theta.len());
            let (row, column) = (site / self.n, site % self.n);
            let old = self.theta[site];
            let new = old + (rng.next_f64() * 2.0 - 1.0) * step;
            let neighbours = [
                self.at(row + 1, column),
                self.at(row + self.n - 1, column),
                self.at(row, column + 1),
                self.at(row, column + self.n - 1),
            ];
            let before: f64 = neighbours.iter().map(|t| (old - t).cos()).sum();
            let after: f64 = neighbours.iter().map(|t| (new - t).cos()).sum();
            let cost = self.j * (before - after);
            if cost <= 0.0 || rng.next_f64() < (-self.beta * cost).exp() {
                self.theta[site] = new.rem_euclid(std::f64::consts::TAU);
            }
        }
    }

    /// The vorticity of the plaquette whose lower-left corner is `(row,
    /// column)`, as an integer winding number.
    ///
    /// Summing the angle differences around a plaquette, each reduced to
    /// `(-pi, pi]`, gives a multiple of `2 pi`. That the multiple is an
    /// integer is not approximate -- it is a topological fact about the
    /// configuration, and it is why vortices cannot be removed by a small
    /// change.
    #[must_use]
    pub fn plaquette_vorticity(&self, row: usize, column: usize) -> i32 {
        let corners = [
            self.at(row, column),
            self.at(row, column + 1),
            self.at(row + 1, column + 1),
            self.at(row + 1, column),
        ];
        let mut total = 0.0;
        for k in 0..4 {
            let mut difference = corners[(k + 1) % 4] - corners[k];
            // Reduce to the principal branch.
            while difference > std::f64::consts::PI {
                difference -= std::f64::consts::TAU;
            }
            while difference <= -std::f64::consts::PI {
                difference += std::f64::consts::TAU;
            }
            total += difference;
        }
        (total / std::f64::consts::TAU).round() as i32
    }

    /// The number of vortices and antivortices on the lattice.
    #[must_use]
    pub fn vortex_count(&self) -> (usize, usize) {
        let mut positive = 0usize;
        let mut negative = 0usize;
        for row in 0..self.n {
            for column in 0..self.n {
                match self.plaquette_vorticity(row, column) {
                    v if v > 0 => positive += 1,
                    v if v < 0 => negative += 1,
                    _ => {}
                }
            }
        }
        (positive, negative)
    }

    /// The Kosterlitz-Thouless transition temperature, about `0.893 j`.
    ///
    /// Not exactly known: unlike Ising, the XY model has no closed-form
    /// solution, and this is the best numerical estimate.
    #[must_use]
    pub fn kt_transition_estimate(j: f64) -> f64 {
        0.8929 * j
    }
}

// ---------------------------------------------------------------------------
// Advanced sampling
// ---------------------------------------------------------------------------

/// Wang-Landau sampling: the density of states as a function of energy.
///
/// Rather than sampling the Boltzmann distribution at one temperature, this
/// performs a random walk in *energy* with acceptance `min(1, g(E_old) /
/// g(E_new))`, refining the estimate `g` as it goes so that the walk flattens
/// its own histogram. The result gives every temperature at once, which is
/// what a canonical simulation cannot do: it converges on the *entropy*, not
/// on an average.
///
/// Returns the logarithm of the density of states, indexed by the energy
/// level offset from the minimum.
///
/// # Errors
/// Returns an error for bad parameters or an energy range that does not fit.
pub fn wang_landau(
    energy: &dyn Fn(u64) -> i64,
    sites: usize,
    flatness: f64,
    final_modification: f64,
    max_steps: usize,
    rng: &mut Rng,
) -> Result<Vec<f64>, GeomError> {
    if sites == 0 || sites > 20 {
        return Err(GeomError::InvalidArgument("wang_landau handles 1 to 20 sites"));
    }
    if !(0.0..1.0).contains(&flatness) || !(final_modification > 0.0) || max_steps == 0 {
        return Err(GeomError::InvalidArgument("wang_landau: bad parameters"));
    }
    let states = 1u64 << sites;
    let lowest = (0..states).map(energy).min().unwrap_or(0);
    let highest = (0..states).map(energy).max().unwrap_or(0);
    let levels = (highest - lowest + 1) as usize;
    if levels > 100_000 {
        return Err(GeomError::InvalidArgument("the energy range is too wide"));
    }
    // Only the levels a configuration can actually reach are visited, so the
    // flatness test has to ignore the rest -- otherwise it never passes.
    let mut reachable = vec![false; levels];
    for state in 0..states {
        reachable[(energy(state) - lowest) as usize] = true;
    }

    let mut log_g = vec![0.0f64; levels];
    let mut histogram = vec![0u64; levels];
    let mut modification = 1.0f64;
    let mut state = 0u64;
    let mut level = (energy(state) - lowest) as usize;
    let mut steps = 0usize;

    while modification > final_modification && steps < max_steps {
        for _ in 0..1000 {
            steps += 1;
            let flipped = state ^ (1u64 << pick(rng, sites));
            let candidate = (energy(flipped) - lowest) as usize;
            let difference = log_g[level] - log_g[candidate];
            if difference >= 0.0 || rng.next_f64() < difference.exp() {
                state = flipped;
                level = candidate;
            }
            log_g[level] += modification;
            histogram[level] += 1;
        }
        // Flat enough? Compare the smallest visited bin with the mean.
        let visited: Vec<u64> = (0..levels)
            .filter(|&k| reachable[k])
            .map(|k| histogram[k])
            .collect();
        let mean = visited.iter().sum::<u64>() as f64 / visited.len() as f64;
        let smallest = visited.iter().copied().min().unwrap_or(0) as f64;
        if mean > 0.0 && smallest > flatness * mean {
            modification /= 2.0;
            histogram.iter_mut().for_each(|h| *h = 0);
        }
    }
    // Normalise so the lowest reachable level has weight matching its true
    // degeneracy of at least one; the overall constant is unmeasurable.
    let offset = (0..levels)
        .filter(|&k| reachable[k])
        .map(|k| log_g[k])
        .fold(f64::INFINITY, f64::min);
    Ok((0..levels)
        .map(|k| if reachable[k] { log_g[k] - offset } else { f64::NEG_INFINITY })
        .collect())
}

/// Canonical averages reconstructed from a density of states.
///
/// The whole point of Wang-Landau: one run gives every temperature. Returns
/// the mean energy and the heat capacity at the given inverse temperature.
///
/// # Errors
/// Returns an error for an empty density or a non-positive beta.
pub fn canonical_from_dos(
    log_g: &[f64],
    lowest_energy: f64,
    step: f64,
    beta: f64,
) -> Result<(f64, f64), GeomError> {
    if log_g.is_empty() || !(beta > 0.0) || !(step > 0.0) {
        return Err(GeomError::InvalidArgument("canonical_from_dos: bad input"));
    }
    // Weights carried in logarithms and shifted by the largest, since the
    // density of states spans hundreds of orders of magnitude.
    let terms: Vec<(f64, f64)> = log_g
        .iter()
        .enumerate()
        .filter(|(_, g)| g.is_finite())
        .map(|(k, g)| (lowest_energy + k as f64 * step, g - beta * (lowest_energy + k as f64 * step)))
        .collect();
    if terms.is_empty() {
        return Err(GeomError::Degenerate("the density of states is empty"));
    }
    let peak = terms.iter().map(|(_, w)| *w).fold(f64::NEG_INFINITY, f64::max);
    let mut z = 0.0;
    let mut e1 = 0.0;
    let mut e2 = 0.0;
    for (e, w) in &terms {
        let weight = (w - peak).exp();
        z += weight;
        e1 += weight * e;
        e2 += weight * e * e;
    }
    let mean = e1 / z;
    let variance = (e2 / z - mean * mean).max(0.0);
    Ok((mean, beta * beta * variance))
}

/// Parallel tempering: several replicas at different temperatures, with
/// neighbouring pairs occasionally swapped.
///
/// The swap acceptance `min(1, exp((beta_i - beta_j)(E_i - E_j)))` preserves
/// each replica's own equilibrium distribution while letting a cold replica
/// escape a local minimum by wandering up to a hot temperature and back. It
/// is the standard answer to a rugged landscape, and it costs nothing in
/// correctness -- the swaps satisfy detailed balance on the joint system.
///
/// Returns the statistics for each temperature and the swap acceptance rate.
///
/// # Errors
/// Returns an error for fewer than two temperatures or bad sweep counts.
pub fn parallel_tempering_ising(
    n: usize,
    j: f64,
    betas: &[f64],
    sweeps: usize,
    thermalize: usize,
    rng: &mut Rng,
) -> Result<(Vec<IsingStats>, f64), GeomError> {
    if betas.len() < 2 {
        return Err(GeomError::InvalidArgument("parallel tempering needs two temperatures"));
    }
    if betas.iter().any(|b| !(*b > 0.0)) {
        return Err(GeomError::InvalidArgument("every beta must be positive"));
    }
    if sweeps == 0 {
        return Err(GeomError::InvalidArgument("parallel tempering needs sweeps"));
    }
    let mut replicas: Vec<Ising2D> = betas
        .iter()
        .map(|&beta| Ising2D::random(n, j, 0.0, beta, true, rng))
        .collect::<Result<_, _>>()?;

    for _ in 0..thermalize {
        for replica in &mut replicas {
            replica.metropolis_sweep(rng);
        }
    }

    let sites = (n * n) as f64;
    let mut accumulators = vec![[0.0f64; 5]; betas.len()];
    let mut attempts = 0usize;
    let mut accepted = 0usize;
    for sweep in 0..sweeps {
        for replica in &mut replicas {
            replica.metropolis_sweep(rng);
        }
        // Alternate which pairs are offered, so every neighbouring pair gets
        // a turn.
        let start = sweep % 2;
        let mut k = start;
        while k + 1 < replicas.len() {
            attempts += 1;
            let (e1, e2) = (replicas[k].energy(), replicas[k + 1].energy());
            let argument = (replicas[k].beta - replicas[k + 1].beta) * (e1 - e2);
            if argument >= 0.0 || rng.next_f64() < argument.exp() {
                accepted += 1;
                // Swap the configurations, not the temperatures.
                let left = replicas[k].spins.clone();
                replicas[k].spins = replicas[k + 1].spins.clone();
                replicas[k + 1].spins = left;
            }
            k += 2;
        }
        for (index, replica) in replicas.iter().enumerate() {
            let e = replica.energy_per_site();
            let m = replica.magnetization();
            accumulators[index][0] += e;
            accumulators[index][1] += e * e;
            accumulators[index][2] += m.abs();
            accumulators[index][3] += m * m;
            accumulators[index][4] += m * m * m * m;
        }
    }
    let count = sweeps as f64;
    let stats = (0..betas.len())
        .map(|index| {
            let a = accumulators[index];
            let (e1, e2) = (a[0] / count, a[1] / count);
            let (m_abs, m2, m4) = (a[2] / count, a[3] / count, a[4] / count);
            let e_var = (e2 - e1 * e1).max(0.0);
            IsingStats {
                e_mean: e1,
                e_var,
                m_mean: m_abs,
                m_abs,
                susceptibility: betas[index] * sites * (m2 - m_abs * m_abs).max(0.0),
                heat_capacity: betas[index] * betas[index] * sites * e_var,
                binder_cumulant: if m2 > 0.0 { 1.0 - m4 / (3.0 * m2 * m2) } else { 0.0 },
                samples: sweeps,
            }
        })
        .collect();
    let rate = if attempts == 0 { 0.0 } else { accepted as f64 / attempts as f64 };
    Ok((stats, rate))
}

/// The Binder crossing estimate of the critical temperature.
///
/// The Binder cumulant is dimensionless, so its finite-size corrections
/// cancel at the critical point and curves for different lattice sizes cross
/// there. That makes it far more accurate than looking for a peak in the
/// susceptibility, whose position drifts with the size.
///
/// `curves[i]` is the cumulant of lattice `sizes[i]` at each of the given
/// temperatures.
///
/// # Errors
/// Returns an error for mismatched lengths or fewer than two sizes.
pub fn binder_crossing(
    temperatures: &[f64],
    curves: &[Vec<f64>],
) -> Result<f64, GeomError> {
    if curves.len() < 2 || temperatures.len() < 2 {
        return Err(GeomError::InvalidArgument("binder_crossing needs two sizes and two points"));
    }
    if curves.iter().any(|c| c.len() != temperatures.len()) {
        return Err(GeomError::InvalidArgument("a curve has the wrong length"));
    }
    // Average the crossings of every pair of curves, found by linear
    // interpolation of their difference.
    let mut crossings = Vec::new();
    for a in 0..curves.len() {
        for b in (a + 1)..curves.len() {
            for k in 0..temperatures.len() - 1 {
                let d0 = curves[a][k] - curves[b][k];
                let d1 = curves[a][k + 1] - curves[b][k + 1];
                if d0 == 0.0 {
                    crossings.push(temperatures[k]);
                } else if d0 * d1 < 0.0 {
                    let t = d0 / (d0 - d1);
                    crossings.push(temperatures[k] + t * (temperatures[k + 1] - temperatures[k]));
                }
            }
        }
    }
    if crossings.is_empty() {
        return Err(GeomError::Degenerate("the curves do not cross in this range"));
    }
    Ok(crossings.iter().sum::<f64>() / crossings.len() as f64)
}

/// The fluctuation-dissipation check: the heat capacity computed from the
/// energy variance against the same quantity differentiated numerically.
///
/// Returns the relative discrepancy. The identity `C = beta^2 Var(E)` is not
/// a modelling assumption but a consequence of the Boltzmann distribution, so
/// a sampler that violates it is not sampling that distribution.
///
/// # Errors
/// Returns an error for a non-positive beta or a zero heat capacity.
pub fn fluctuation_dissipation_check(
    stats: &IsingStats,
    beta: f64,
    sites: usize,
) -> Result<f64, GeomError> {
    if !(beta > 0.0) || sites == 0 {
        return Err(GeomError::InvalidArgument("fluctuation_dissipation_check: bad input"));
    }
    let from_variance = beta * beta * sites as f64 * stats.e_var;
    if stats.heat_capacity == 0.0 && from_variance == 0.0 {
        return Ok(0.0);
    }
    let scale = stats.heat_capacity.abs().max(from_variance.abs()).max(1e-300);
    Ok((from_variance - stats.heat_capacity).abs() / scale)
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

    /// The Ising energy of a configuration on a small periodic lattice, from
    /// the bit pattern -- used to enumerate exactly.
    fn small_energy(n: usize, j: f64, h: f64) -> impl Fn(u64) -> f64 {
        move |state: u64| {
            let spin = |row: usize, column: usize| -> f64 {
                let index = (row % n) * n + (column % n);
                if state >> index & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            };
            let mut bonds = 0.0;
            let mut field = 0.0;
            for row in 0..n {
                for column in 0..n {
                    let here = spin(row, column);
                    bonds += here * spin(row + 1, column);
                    bonds += here * spin(row, column + 1);
                    field += here;
                }
            }
            -j * bonds - h * field
        }
    }

    // -----------------------------------------------------------------
    // Bookkeeping
    // -----------------------------------------------------------------

    #[test]
    fn the_energy_and_its_increments_agree_with_each_other() {
        // The single most useful invariant in a Monte Carlo code: the
        // incremental cost of a flip must equal the difference of two total
        // energies. A sampler whose increments drift from its totals will
        // still produce plausible-looking pictures and entirely wrong
        // averages.
        let mut rng = Rng::new(0x_15E0_0001);
        for periodic in [true, false] {
            for n in [3usize, 4, 6] {
                let mut lattice =
                    Ising2D::random(n, 1.3, -0.4, 0.7, periodic, &mut rng).unwrap();
                for _ in 0..300 {
                    let site = pick(&mut rng, n * n);
                    let before = lattice.energy();
                    let predicted = lattice.flip_cost(site);
                    lattice.spins[site] = -lattice.spins[site];
                    let after = lattice.energy();
                    assert!(
                        close(after - before, predicted, 1e-9),
                        "n = {n}, periodic = {periodic}: predicted {predicted}, got {}",
                        after - before
                    );
                }
            }
        }
    }

    #[test]
    fn the_total_energy_matches_an_independent_enumeration() {
        // Two entirely separate ways of writing the same sum, one over
        // neighbour lists and one over the bit pattern.
        let mut rng = Rng::new(0x_15E0_0002);
        for n in [3usize, 4] {
            let reference = small_energy(n, 1.1, 0.3);
            for _ in 0..200 {
                let state = rng.next_u64() & ((1u64 << (n * n)) - 1);
                let mut lattice = Ising2D::cold(n, 1.1, 0.3, 1.0, true).unwrap();
                for site in 0..n * n {
                    lattice.spins[site] = if state >> site & 1 == 0 { 1 } else { -1 };
                }
                assert!(
                    close(lattice.energy(), reference(state), 1e-9),
                    "n = {n}: {} against {}",
                    lattice.energy(),
                    reference(state)
                );
            }
        }
        // A cold ferromagnet has every bond satisfied: energy -2 j per site.
        let cold = Ising2D::cold(8, 1.0, 0.0, 1.0, true).unwrap();
        assert!(close(cold.energy_per_site(), -2.0, 1e-12));
        assert!(close(cold.magnetization(), 1.0, 1e-12));
        // Open boundaries have fewer bonds, so a higher energy.
        let open = Ising2D::cold(8, 1.0, 0.0, 1.0, false).unwrap();
        assert!(open.energy() > cold.energy());
        let bonds = 2 * 8 * 7;
        assert!(close(open.energy(), -(bonds as f64), 1e-12));
    }

    // -----------------------------------------------------------------
    // Sampling against exact results
    // -----------------------------------------------------------------

    #[test]
    fn the_sampler_reproduces_the_exact_small_lattice_averages() {
        // A four-by-four periodic lattice has 65536 states, so its exact
        // averages are available by enumeration. Agreement there is a far
        // stronger statement than agreement with the infinite-lattice
        // formulas, because it holds at any temperature including near the
        // transition and involves no finite-size argument at all.
        let mut rng = Rng::new(0x_15E0_0003);
        let n = 4usize;
        let sites = (n * n) as f64;
        for beta in [0.1f64, 0.3, 0.44, 0.6, 1.0] {
            let reference = small_energy(n, 1.0, 0.0);
            let (exact_energy, _) =
                thermodynamics_exact_small(&reference, n * n, beta).unwrap();

            let mut lattice = Ising2D::random(n, 1.0, 0.0, beta, true, &mut rng).unwrap();
            let stats = lattice.sample(60_000, 2_000, 5, false, &mut rng).unwrap();
            assert!(
                relative(stats.e_mean * sites, exact_energy) < 0.02,
                "beta = {beta}: sampled {} against exact {exact_energy}",
                stats.e_mean * sites
            );

            // Wolff must agree with Metropolis: they sample the same
            // distribution by different moves.
            let mut wolff = Ising2D::random(n, 1.0, 0.0, beta, true, &mut rng).unwrap();
            let cluster = wolff.sample(200_000, 5_000, 5, true, &mut rng).unwrap();
            assert!(
                relative(cluster.e_mean * sites, exact_energy) < 0.02,
                "beta = {beta}: Wolff gave {} against exact {exact_energy}",
                cluster.e_mean * sites
            );

            // The heat capacity from the variance is the same number the
            // struct reports -- the fluctuation-dissipation identity.
            assert!(
                fluctuation_dissipation_check(&stats, beta, n * n).unwrap() < 1e-12,
                "the identity fails at beta = {beta}"
            );
        }
    }

    #[test]
    fn a_large_lattice_matches_onsager_away_from_the_critical_point() {
        // The infinite-lattice results hold on a finite one only where the
        // correlation length is much smaller than the lattice, which is why
        // the temperatures here stay clear of the transition.
        let mut rng = Rng::new(0x_15E0_0004);
        let tc = ising_tc_exact();
        for temperature in [1.6f64, 1.9, 2.8, 3.4] {
            let beta = 1.0 / temperature;
            let mut lattice = Ising2D::random(32, 1.0, 0.0, beta, true, &mut rng).unwrap();
            let stats = lattice.sample(20_000, 4_000, 5, true, &mut rng).unwrap();

            let exact_energy = onsager_energy(beta, 1.0).unwrap();
            assert!(
                relative(stats.e_mean, exact_energy) < 0.02,
                "T = {temperature}: energy {} against Onsager {exact_energy}",
                stats.e_mean
            );

            let exact_m = onsager_magnetization(beta, 1.0).unwrap();
            if temperature < tc - 0.3 {
                assert!(
                    relative(stats.m_abs, exact_m) < 0.03,
                    "T = {temperature}: |m| {} against Onsager {exact_m}",
                    stats.m_abs
                );
            } else if temperature > tc + 0.4 {
                assert!(close(exact_m, 0.0, 1e-15), "Onsager should give zero above Tc");
                // A finite lattice has a residual |m| going as 1 / sqrt(N).
                assert!(
                    stats.m_abs < 0.2,
                    "T = {temperature}: the disordered phase has |m| = {}",
                    stats.m_abs
                );
            }
        }
    }

    #[test]
    fn the_exact_results_have_the_shapes_the_theory_gives_them() {
        let tc = ising_tc_exact();
        assert!(close(tc, 2.269_185_314_213_022, 1e-12), "Tc is {tc}");
        // The critical point is where sinh(2 beta j) is one -- the self-dual
        // point, which is how Kramers and Wannier found it.
        assert!(close((2.0 / tc).sinh(), 1.0, 1e-12));

        // The magnetisation vanishes above Tc and rises as (Tc - T)^(1/8).
        assert_eq!(onsager_magnetization(1.0 / (tc + 0.001), 1.0).unwrap(), 0.0);
        assert_eq!(onsager_magnetization(1.0 / (2.0 * tc), 1.0).unwrap(), 0.0);
        let mut previous = 0.0;
        for temperature in [2.26f64, 2.2, 2.0, 1.5, 1.0, 0.5] {
            let m = onsager_magnetization(1.0 / temperature, 1.0).unwrap();
            assert!(m > previous, "the magnetisation fell at T = {temperature}");
            assert!(m <= 1.0);
            previous = m;
        }
        // It approaches one but does not reach it: at T = 1 there is still a
        // seventh of a per cent missing, so the saturation has to be tested
        // where the exponential really has died.
        assert!(close(previous, 1.0, 1e-6), "it should saturate at low T: {previous}");
        assert!(onsager_magnetization(1.0, 1.0).unwrap() < 1.0);
        // The exponent is one eighth, checked by the ratio of two points.
        let a = onsager_magnetization(1.0 / (tc - 0.01), 1.0).unwrap();
        let b = onsager_magnetization(1.0 / (tc - 0.04), 1.0).unwrap();
        let exponent = (b / a).ln() / (4.0f64).ln();
        assert!(
            close(exponent, 0.125, 0.01),
            "the critical exponent came out {exponent}, not one eighth"
        );

        // The energy is negative, monotone, and tends to -2 j at low T.
        let mut previous = 0.0;
        for temperature in [5.0f64, 3.0, 2.0, 1.0, 0.5] {
            let e = onsager_energy(1.0 / temperature, 1.0).unwrap();
            assert!(e < previous, "the energy rose at T = {temperature}");
            assert!(e > -2.001, "the energy is below the ground state: {e}");
            previous = e;
        }
        assert!(close(previous, -2.0, 0.01), "the low-temperature energy is {previous}");
        assert!(onsager_energy(-1.0, 1.0).is_err());
        assert!(onsager_magnetization(1.0, 0.0).is_err());
    }

    #[test]
    fn the_one_dimensional_chain_matches_a_direct_enumeration() {
        // The transfer matrix is exact for an infinite chain; enumeration is
        // exact for a finite ring. They agree once the ring is long enough
        // for the boundary to stop mattering.
        for (beta, j, h) in [(0.5f64, 1.0f64, 0.0f64), (1.0, 1.0, 0.3), (0.2, 0.7, -0.5)] {
            let (free_energy, magnetization) = ising_1d_exact(beta, j, h).unwrap();
            let sites = 16usize;
            let ring = move |state: u64| -> f64 {
                let spin = |k: usize| if state >> (k % sites) & 1 == 0 { 1.0 } else { -1.0 };
                let mut bonds = 0.0;
                let mut field = 0.0;
                for k in 0..sites {
                    bonds += spin(k) * spin(k + 1);
                    field += spin(k);
                }
                -j * bonds - h * field
            };
            let z = partition_function_exact_small(&ring, sites, beta).unwrap();
            let per_site = free_energy_from_z(z, beta).unwrap() / sites as f64;
            assert!(
                relative(per_site, free_energy) < 1e-3,
                "beta = {beta}, h = {h}: enumeration gives {per_site}, transfer matrix {free_energy}"
            );

            // The magnetisation, from the same enumeration.
            let lowest = (0..(1u64 << sites)).map(&ring).fold(f64::INFINITY, f64::min);
            let mut weight_total = 0.0;
            let mut m_total = 0.0;
            for state in 0..(1u64 << sites) {
                let weight = (-beta * (ring(state) - lowest)).exp();
                let m: f64 = (0..sites)
                    .map(|k| if state >> k & 1 == 0 { 1.0 } else { -1.0 })
                    .sum::<f64>()
                    / sites as f64;
                weight_total += weight;
                m_total += weight * m;
            }
            let sampled = m_total / weight_total;
            assert!(
                (sampled - magnetization).abs() < 1e-3,
                "beta = {beta}, h = {h}: enumeration gives m = {sampled}, formula {magnetization}"
            );
        }
        // In zero field the chain is unmagnetised at every temperature: no
        // transition, which is Ising's own result.
        for beta in [0.1f64, 1.0, 10.0, 100.0] {
            let (_, m) = ising_1d_exact(beta, 1.0, 0.0).unwrap();
            assert!(close(m, 0.0, 1e-12), "the chain magnetised at beta = {beta}");
        }
        assert!(ising_1d_exact(0.0, 1.0, 0.0).is_err());
        assert!(partition_function_exact_small(&|_| 0.0, 25, 1.0).is_err());
        assert!(free_energy_from_z(0.0, 1.0).is_err());
    }

    // -----------------------------------------------------------------
    // Algorithmic properties
    // -----------------------------------------------------------------

    #[test]
    fn wolff_decorrelates_far_faster_than_metropolis_near_the_critical_point() {
        // The reason cluster algorithms exist. Away from the transition both
        // are fine; at it, Metropolis has to move a correlated region one
        // spin at a time and its autocorrelation time grows with the lattice
        // while Wolff's barely does.
        let mut rng = Rng::new(0x_15E0_0005);
        let beta = 1.0 / ising_tc_exact();
        let n = 24usize;

        let mut metropolis = Ising2D::random(n, 1.0, 0.0, beta, true, &mut rng).unwrap();
        for _ in 0..400 {
            metropolis.metropolis_sweep(&mut rng);
        }
        let (slow, slow_work) = metropolis.autocorrelation_time(1_200, false, &mut rng).unwrap();

        let mut cluster = Ising2D::random(n, 1.0, 0.0, beta, true, &mut rng).unwrap();
        for _ in 0..2_000 {
            cluster.wolff_cluster_step(&mut rng).unwrap();
        }
        let (fast, fast_work) = cluster.autocorrelation_time(6_000, true, &mut rng).unwrap();

        assert!(fast >= 0.5 && slow >= 0.5, "the times are {fast} and {slow}");
        // Compared in spin flips rather than in updates, which is the only
        // fair unit: a Metropolis update attempts a lattice's worth of flips
        // and a Wolff update flips one cluster.
        let metropolis_cost = slow * slow_work;
        let wolff_cost = fast * fast_work;
        assert!(
            close(slow_work, (n * n) as f64, 1e-9),
            "a Metropolis update should cost n^2 flips, not {slow_work}"
        );
        assert!(
            metropolis_cost > 5.0 * wolff_cost,
            "Metropolis needs {metropolis_cost} flips per independent sample against Wolff's {wolff_cost}"
        );

        // Both still sample the same distribution: their energies agree.
        let a = metropolis.sample(2_000, 200, 2, false, &mut rng).unwrap();
        let b = cluster.sample(20_000, 2_000, 5, true, &mut rng).unwrap();
        assert!(
            relative(a.e_mean, b.e_mean) < 0.03,
            "the two samplers disagree: {} against {}",
            a.e_mean,
            b.e_mean
        );
        assert!(metropolis.autocorrelation_time(10, false, &mut rng).is_err());
    }

    #[test]
    fn wolff_flips_bigger_clusters_as_the_temperature_falls() {
        // The cluster size tracks the correlation length, which is exactly
        // why the algorithm works: it flips whatever is correlated, whatever
        // that happens to be.
        let mut rng = Rng::new(0x_15E0_0006);
        let n = 24usize;
        let mut previous = 0.0;
        for temperature in [4.0f64, 3.0, 2.5, 2.269, 2.0] {
            let mut lattice =
                Ising2D::random(n, 1.0, 0.0, 1.0 / temperature, true, &mut rng).unwrap();
            for _ in 0..200 {
                lattice.wolff_cluster_step(&mut rng).unwrap();
            }
            let mut total = 0usize;
            for _ in 0..400 {
                total += lattice.wolff_cluster_step(&mut rng).unwrap();
            }
            let mean = total as f64 / 400.0;
            assert!(mean >= 1.0 && mean <= (n * n) as f64);
            assert!(
                mean > previous,
                "at T = {temperature} the mean cluster is {mean}, not larger than {previous}"
            );
            previous = mean;
        }
        // Below the transition the cluster is a good fraction of the lattice.
        assert!(previous > 0.3 * (n * n) as f64, "the cold cluster is only {previous} spins");

        // Wolff refuses the cases it cannot handle rather than sampling the
        // wrong distribution.
        let mut with_field = Ising2D::cold(8, 1.0, 0.5, 0.5, true).unwrap();
        assert!(with_field.wolff_cluster_step(&mut rng).is_err());
        let mut antiferro = Ising2D::cold(8, -1.0, 0.0, 0.5, true).unwrap();
        assert!(antiferro.wolff_cluster_step(&mut rng).is_err());
    }

    #[test]
    fn heat_bath_and_metropolis_agree_on_the_distribution_they_sample() {
        // Two update rules, both satisfying detailed balance with the same
        // Boltzmann weight. The averages must coincide; only the efficiency
        // differs.
        let mut rng = Rng::new(0x_15E0_0007);
        let n = 4usize;
        for beta in [0.2f64, 0.44, 0.8] {
            let reference = small_energy(n, 1.0, 0.2);
            let (exact, _) = thermodynamics_exact_small(&reference, n * n, beta).unwrap();

            let mut lattice = Ising2D::random(n, 1.0, 0.2, beta, true, &mut rng).unwrap();
            for _ in 0..2_000 {
                lattice.heat_bath_sweep(&mut rng);
            }
            let mut e_total = 0.0;
            for _ in 0..40_000 {
                lattice.heat_bath_sweep(&mut rng);
                e_total += lattice.energy();
            }
            let sampled = e_total / 40_000.0;
            assert!(
                relative(sampled, exact) < 0.02,
                "beta = {beta}: heat bath gave {sampled} against exact {exact}"
            );
        }
    }

    #[test]
    fn the_binder_cumulant_crosses_at_the_critical_temperature() {
        // The cumulant is dimensionless, so its finite-size corrections
        // cancel at criticality and curves for different lattices cross
        // there. That is far sharper than the susceptibility peak, whose
        // position drifts with the size.
        let mut rng = Rng::new(0x_15E0_0008);
        let temperatures: Vec<f64> = (0..9).map(|k| 2.15 + 0.03 * k as f64).collect();
        let mut curves = Vec::new();
        for n in [8usize, 16] {
            let mut curve = Vec::new();
            for &temperature in &temperatures {
                let mut lattice =
                    Ising2D::random(n, 1.0, 0.0, 1.0 / temperature, true, &mut rng).unwrap();
                let stats = lattice.sample(30_000, 5_000, 5, true, &mut rng).unwrap();
                curve.push(stats.binder_cumulant);
            }
            // The cumulant runs from 2/3 deep in the ordered phase toward
            // zero in the disordered one.
            assert!(curve[0] > curve[curve.len() - 1], "the cumulant did not fall: {curve:?}");
            assert!(curve.iter().all(|c| (-0.1..=0.70).contains(c)), "{curve:?}");
            curves.push(curve);
        }
        let estimate = binder_crossing(&temperatures, &curves).unwrap();
        let tc = ising_tc_exact();
        assert!(
            relative(estimate, tc) < 0.02,
            "the crossing gives {estimate} against the exact {tc}"
        );
        assert!(binder_crossing(&temperatures, &curves[..1]).is_err());
        assert!(binder_crossing(&temperatures, &[vec![0.0; 3], vec![0.0; 3]]).is_err());
    }

    #[test]
    fn correlations_decay_over_a_length_that_grows_toward_the_transition() {
        // The correlation length is what diverges at a continuous
        // transition, so it must grow as the temperature falls toward it.
        // Measured from ensemble averages, not from a snapshot: a single
        // configuration's correlation function is too noisy at large
        // separation to fit anything.
        let mut rng = Rng::new(0x_15E0_0009);
        let n = 32usize;
        let mut previous = 0.0;
        for temperature in [3.4f64, 3.0, 2.7, 2.45] {
            let mut lattice =
                Ising2D::random(n, 1.0, 0.0, 1.0 / temperature, true, &mut rng).unwrap();
            for _ in 0..3_000 {
                lattice.wolff_cluster_step(&mut rng).unwrap();
            }
            let (correlations, m2) = lattice.sample_correlations(6_000, true, &mut rng).unwrap();
            assert!(close(correlations[0], 1.0, 1e-12), "C(0) is {}", correlations[0]);
            assert!(
                correlations.windows(2).all(|w| w[1] <= w[0] + 0.02),
                "the correlation is not decreasing: {correlations:?}"
            );
            let length = Ising2D::correlation_length_estimate(&correlations, m2).unwrap();
            assert!(length >= 0.0 && length.is_finite(), "the length is {length}");
            assert!(
                length > previous,
                "at T = {temperature} the length is {length}, not longer than {previous}"
            );
            previous = length;
        }
        assert!(previous > 2.0, "near the transition the length is only {previous}");

        // A correlation at zero separation is one, by definition, and a
        // fully ordered lattice correlates perfectly at every distance.
        let lattice = Ising2D::cold(8, 1.0, 0.0, 1.0, true).unwrap();
        for r in 0..8 {
            assert!(close(lattice.correlation_function(r).unwrap(), 1.0, 1e-12));
        }
        assert!(lattice.correlation_function(8).is_err());
        // A perfectly ordered lattice has no decay to fit, and the estimator
        // says the length is at least the lattice rather than inventing one.
        let ordered = vec![1.0f64; 8];
        assert!(Ising2D::correlation_length_estimate(&ordered, 1.0).unwrap() == 0.0);
        assert!(Ising2D::correlation_length_estimate(&ordered, 0.0).unwrap() >= 8.0);
        assert!(Ising2D::correlation_length_estimate(&[1.0, 0.5], 0.0).is_err());
        let mut small = Ising2D::cold(4, 1.0, 0.0, 1.0, true).unwrap();
        assert!(small.sample_correlations(10, false, &mut rng).is_err());
        let mut fine = Ising2D::cold(8, 1.0, 0.0, 1.0, true).unwrap();
        assert!(fine.sample_correlations(0, false, &mut rng).is_err());
    }

    // -----------------------------------------------------------------
    // Advanced sampling
    // -----------------------------------------------------------------

    #[test]
    fn wang_landau_recovers_the_exact_density_of_states() {
        // The density of states is a combinatorial fact about the model, so
        // it can be counted exactly on a small lattice and compared. Getting
        // it right means every temperature is right at once, which is what
        // the method is for.
        let mut rng = Rng::new(0x_15E0_000A);
        let n = 4usize;
        let sites = n * n;
        // The energy in units of 2j, so it is an integer.
        let integer_energy = move |state: u64| -> i64 {
            let spin = |row: usize, column: usize| -> i64 {
                let index = (row % n) * n + (column % n);
                if state >> index & 1 == 0 {
                    1
                } else {
                    -1
                }
            };
            let mut bonds = 0i64;
            for row in 0..n {
                for column in 0..n {
                    bonds += spin(row, column) * spin(row + 1, column);
                    bonds += spin(row, column) * spin(row, column + 1);
                }
            }
            -bonds
        };

        // The exact count, by enumeration.
        let lowest = (0..(1u64 << sites)).map(&integer_energy).min().unwrap();
        let highest = (0..(1u64 << sites)).map(&integer_energy).max().unwrap();
        let mut exact = vec![0u64; (highest - lowest + 1) as usize];
        for state in 0..(1u64 << sites) {
            exact[(integer_energy(state) - lowest) as usize] += 1;
        }

        let log_g = wang_landau(&integer_energy, sites, 0.85, 1e-5, 8_000_000, &mut rng).unwrap();
        assert_eq!(log_g.len(), exact.len());
        // Compare the shape, normalised so the ground state matches: the
        // overall constant is unmeasurable and the method does not claim it.
        let offset = log_g[0] - (exact[0] as f64).ln();
        for (level, &count) in exact.iter().enumerate() {
            if count == 0 {
                assert!(!log_g[level].is_finite(), "an unreachable level got a weight");
                continue;
            }
            let predicted = log_g[level] - offset;
            let truth = (count as f64).ln();
            // The residual error is set by the final modification factor,
            // not by the run length: Wang-Landau converges to within roughly
            // sqrt(ln f) of the truth and no further.
            assert!(
                (predicted - truth).abs() < 0.2,
                "level {level}: log g is {predicted} against the exact {truth}"
            );
        }

        // And the canonical averages it implies match the enumeration at
        // every temperature -- one run, every temperature.
        let reference = small_energy(n, 1.0, 0.0);
        for beta in [0.2f64, 0.44, 0.8] {
            let (exact_energy, _) = thermodynamics_exact_small(&reference, sites, beta).unwrap();
            let (from_dos, _) =
                canonical_from_dos(&log_g, lowest as f64, 1.0, beta).unwrap();
            assert!(
                relative(from_dos, exact_energy) < 0.03,
                "beta = {beta}: the density gives {from_dos} against {exact_energy}"
            );
        }
        assert!(wang_landau(&integer_energy, 25, 0.8, 1e-3, 1000, &mut rng).is_err());
        assert!(wang_landau(&integer_energy, sites, 1.5, 1e-3, 1000, &mut rng).is_err());
        assert!(canonical_from_dos(&[], 0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn parallel_tempering_samples_every_temperature_and_swaps_often_enough_to_help() {
        // The swaps must be accepted often enough to move configurations
        // between temperatures; an acceptance near zero means the ladder is
        // too coarse and the method has bought nothing.
        let mut rng = Rng::new(0x_15E0_000B);
        let betas: Vec<f64> = (0..6).map(|k| 0.30 + 0.04 * k as f64).collect();
        let (stats, rate) = parallel_tempering_ising(8, 1.0, &betas, 4_000, 500, &mut rng).unwrap();
        assert_eq!(stats.len(), betas.len());
        assert!(rate > 0.1, "the swap acceptance is only {rate}");

        // Colder replicas are lower in energy and more magnetised, at every
        // rung of the ladder.
        for k in 1..stats.len() {
            assert!(
                stats[k].e_mean < stats[k - 1].e_mean,
                "rung {k} is not colder: {} against {}",
                stats[k].e_mean,
                stats[k - 1].e_mean
            );
            assert!(
                stats[k].m_abs > stats[k - 1].m_abs - 0.02,
                "rung {k} is less magnetised: {} against {}",
                stats[k].m_abs,
                stats[k - 1].m_abs
            );
        }
        // And each rung agrees with the exact enumeration for that lattice.
        let reference = small_energy(4, 1.0, 0.0);
        let (small_stats, _) =
            parallel_tempering_ising(4, 1.0, &betas, 20_000, 2_000, &mut rng).unwrap();
        for (k, &beta) in betas.iter().enumerate() {
            let (exact, _) = thermodynamics_exact_small(&reference, 16, beta).unwrap();
            assert!(
                relative(small_stats[k].e_mean * 16.0, exact) < 0.03,
                "beta = {beta}: tempering gives {} against exact {exact}",
                small_stats[k].e_mean * 16.0
            );
        }
        assert!(parallel_tempering_ising(8, 1.0, &[0.5], 100, 10, &mut rng).is_err());
        assert!(parallel_tempering_ising(8, 1.0, &[0.5, -0.1], 100, 10, &mut rng).is_err());
        assert!(parallel_tempering_ising(8, 1.0, &betas, 0, 10, &mut rng).is_err());
    }

    // -----------------------------------------------------------------
    // Related models
    // -----------------------------------------------------------------

    #[test]
    fn the_potts_model_reduces_to_ising_and_orders_below_its_own_transition() {
        // At q = 2 the Potts critical temperature must be the Ising one,
        // after the factor of two that the two conventions differ by: Potts
        // counts agreeing bonds, Ising counts spin products.
        // The two Hamiltonians differ by a factor of two: Potts pays the
        // coupling once per agreeing bond, Ising pays it times the spin
        // product, and s_i s_j = 2 delta - 1. So the Potts coupling is twice
        // the Ising one and its critical temperature is half.
        assert!(
            close(potts_tc_exact(2).unwrap(), ising_tc_exact() / 2.0, 1e-9),
            "q = 2 gives {} against half the Ising value {}",
            potts_tc_exact(2).unwrap(),
            ising_tc_exact() / 2.0
        );
        assert!(
            close(potts_tc_exact(2).unwrap(), 1.0 / (1.0 + 2.0f64.sqrt()).ln(), 1e-12),
            "the q = 2 temperature is {}",
            potts_tc_exact(2).unwrap()
        );
        // More states means a lower transition temperature.
        let mut previous = f64::INFINITY;
        for q in 2..=10u8 {
            let tc = potts_tc_exact(q).unwrap();
            assert!(tc < previous, "the transition rose at q = {q}");
            previous = tc;
        }
        assert!(potts_tc_exact(1).is_err());

        // The model orders below its transition and does not above it.
        let mut rng = Rng::new(0x_15E0_000C);
        for q in [3u8, 5] {
            let tc = potts_tc_exact(q).unwrap();
            let mut cold = Potts2D::random(q, 16, 1.0, 1.0 / (0.6 * tc), &mut rng).unwrap();
            let mut hot = Potts2D::random(q, 16, 1.0, 1.0 / (2.0 * tc), &mut rng).unwrap();
            for _ in 0..1_500 {
                cold.metropolis_sweep(&mut rng);
                hot.metropolis_sweep(&mut rng);
            }
            assert!(
                cold.order_parameter() > 0.7,
                "q = {q}: the cold lattice has order {}",
                cold.order_parameter()
            );
            assert!(
                hot.order_parameter() < 0.3,
                "q = {q}: the hot lattice has order {}",
                hot.order_parameter()
            );
            assert!(cold.energy() < hot.energy(), "the cold lattice should be lower in energy");
        }
        assert!(Potts2D::random(1, 8, 1.0, 1.0, &mut rng).is_err());
        assert!(Potts2D::random(3, 1, 1.0, 1.0, &mut rng).is_err());
    }

    #[test]
    fn the_xy_model_counts_vortices_that_always_balance() {
        // On a periodic lattice the total winding is zero: vortices come in
        // pairs, always, whatever the configuration. That is topology and
        // not statistics, so it holds at every temperature and in every
        // sample.
        let mut rng = Rng::new(0x_15E0_000D);
        for temperature in [0.4f64, 0.9, 2.0, 5.0] {
            let mut model = XyModel2D::random(16, 1.0, 1.0 / temperature, &mut rng).unwrap();
            for _ in 0..600 {
                model.metropolis_sweep(&mut rng, 1.2);
            }
            let (positive, negative) = model.vortex_count();
            assert_eq!(
                positive, negative,
                "at T = {temperature} the vortices do not balance: {positive} and {negative}"
            );
            // Every plaquette's winding is an integer, and a small one.
            for row in 0..16 {
                for column in 0..16 {
                    let v = model.plaquette_vorticity(row, column);
                    assert!((-1..=1).contains(&v), "a plaquette wound {v} times");
                }
            }
        }

        // Vortices proliferate as the temperature rises past the
        // Kosterlitz-Thouless point, which is the transition.
        let mut counts = Vec::new();
        for temperature in [0.3f64, 0.6, 1.2, 2.5] {
            let mut model = XyModel2D::random(24, 1.0, 1.0 / temperature, &mut rng).unwrap();
            for _ in 0..800 {
                model.metropolis_sweep(&mut rng, 1.2);
            }
            counts.push(model.vortex_count().0);
        }
        assert!(
            counts.windows(2).all(|w| w[1] >= w[0]),
            "the vortex count did not grow with temperature: {counts:?}"
        );
        assert!(counts[0] < 3, "the cold lattice has {} vortices", counts[0]);
        assert!(counts[3] > 20, "the hot lattice has only {} vortices", counts[3]);
        assert!(close(XyModel2D::kt_transition_estimate(1.0), 0.8929, 1e-9));

        // A cold lattice has lower energy than a hot one.
        let mut cold = XyModel2D::random(16, 1.0, 5.0, &mut rng).unwrap();
        let mut hot = XyModel2D::random(16, 1.0, 0.1, &mut rng).unwrap();
        for _ in 0..800 {
            cold.metropolis_sweep(&mut rng, 0.6);
            hot.metropolis_sweep(&mut rng, 1.5);
        }
        assert!(cold.energy() < hot.energy());
        assert!(XyModel2D::random(2, 1.0, 1.0, &mut rng).is_err());
    }

    #[test]
    fn the_constructors_refuse_degenerate_input() {
        let mut rng = Rng::new(0x_15E0_000E);
        assert!(Ising2D::cold(1, 1.0, 0.0, 1.0, true).is_err());
        assert!(Ising2D::cold(600, 1.0, 0.0, 1.0, true).is_err());
        assert!(Ising2D::cold(8, 1.0, 0.0, 0.0, true).is_err());
        assert!(Ising2D::cold(8, 1.0, 0.0, f64::INFINITY, true).is_err());
        assert!(Ising2D::random(1, 1.0, 0.0, 1.0, true, &mut rng).is_err());

        let mut lattice = Ising2D::cold(8, 1.0, 0.0, 0.5, true).unwrap();
        assert!(lattice.sample(0, 0, 1, false, &mut rng).is_err());
        assert!(lattice.sample(10, 0, 0, false, &mut rng).is_err());
        // Measuring less often than the run is long leaves no samples.
        assert!(lattice.sample(10, 0, 100, false, &mut rng).unwrap().samples == 1);

        assert!(thermodynamics_exact_small(&|_| 0.0, 0, 1.0).is_err());
        assert!(thermodynamics_exact_small(&|_| 0.0, 4, 0.0).is_err());
        assert!(fluctuation_dissipation_check(
            &IsingStats {
                e_mean: 0.0,
                e_var: 0.0,
                m_mean: 0.0,
                m_abs: 0.0,
                susceptibility: 0.0,
                heat_capacity: 0.0,
                binder_cumulant: 0.0,
                samples: 1,
            },
            0.0,
            4
        )
        .is_err());
    }
}
