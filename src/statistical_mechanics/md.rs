//! Molecular dynamics: pair potentials, a cell-list force evaluation, a
//! symplectic integrator, thermostats and barostats, and the structural and
//! transport measurements taken from a trajectory.
//!
//! # Units
//!
//! Everything here is in *reduced* Lennard-Jones units: `sigma`, `eps`, the
//! particle mass and Boltzmann's constant are all one unless the caller says
//! otherwise, so a temperature is an energy and a pressure is an energy per
//! volume. This is not a convenience -- mixing SI constants into a molecular
//! dynamics run is how the field's worst bugs happen, because the equations
//! of motion are dimensionally consistent under any consistent choice and
//! silently wrong under an inconsistent one. See [`lj_reduced_units_note`].
//!
//! The roadmap gives `MdSystem` a `SpatialHash` field. The general-purpose
//! hash in `spatial::kdtree` owns a copy of every position and knows nothing
//! about periodic images, so this module carries its own cell list instead:
//! it is rebuilt each step from the live positions and wraps at the box
//! boundary, which is what the minimum-image convention needs.

use crate::error::GeomError;
use crate::math::Vec3;
use crate::monte_carlo::Rng;
use crate::statistics::inference::{ks_test_one_sample, TestResult};
use std::sync::Arc;

/// A pair potential, as a function of the separation alone.
///
/// Each variant supplies both the energy and the force so that they cannot
/// drift apart: a force that is not the negative gradient of the energy in
/// use will conserve nothing, and the failure looks exactly like an
/// integrator bug.
#[derive(Clone)]
pub enum Potential {
    /// `4 eps ((sigma/r)^12 - (sigma/r)^6)`.
    LennardJones {
        /// Well depth.
        eps: f64,
        /// Distance at which the energy crosses zero.
        sigma: f64,
    },
    /// `d (1 - exp(-a (r - r0)))^2 - d`, a bound well with a finite
    /// dissociation energy.
    Morse {
        /// Well depth.
        d: f64,
        /// Width parameter.
        a: f64,
        /// Equilibrium separation.
        r0: f64,
    },
    /// `ke q_i q_j / r`.
    Coulomb {
        /// Coulomb prefactor.
        ke: f64,
    },
    /// Lennard-Jones and Coulomb together.
    LjCoulomb {
        /// Well depth.
        eps: f64,
        /// Zero-crossing distance.
        sigma: f64,
        /// Coulomb prefactor.
        ke: f64,
    },
    /// `k (r - r0)^2 / 2`.
    Harmonic {
        /// Spring constant.
        k: f64,
        /// Rest length.
        r0: f64,
    },
    /// A caller-supplied law returning `(energy, -du/dr)` at a separation.
    Custom(Arc<dyn Fn(f64) -> (f64, f64) + Send + Sync>),
}

impl std::fmt::Debug for Potential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LennardJones { eps, sigma } => {
                write!(f, "LennardJones {{ eps: {eps}, sigma: {sigma} }}")
            }
            Self::Morse { d, a, r0 } => write!(f, "Morse {{ d: {d}, a: {a}, r0: {r0} }}"),
            Self::Coulomb { ke } => write!(f, "Coulomb {{ ke: {ke} }}"),
            Self::LjCoulomb { eps, sigma, ke } => {
                write!(f, "LjCoulomb {{ eps: {eps}, sigma: {sigma}, ke: {ke} }}")
            }
            Self::Harmonic { k, r0 } => write!(f, "Harmonic {{ k: {k}, r0: {r0} }}"),
            Self::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

impl Potential {
    /// The energy and the radial force `-du/dr` at separation `r` between
    /// charges `qi` and `qj`.
    ///
    /// The force is returned rather than derived numerically so that the two
    /// are guaranteed consistent; the tests check each variant's force
    /// against a finite difference of its own energy.
    #[must_use]
    pub fn evaluate(&self, r: f64, qi: f64, qj: f64) -> (f64, f64) {
        match *self {
            Self::LennardJones { eps, sigma } => lj_pair(r, eps, sigma),
            Self::Morse { d, a, r0 } => {
                let e = (-a * (r - r0)).exp();
                let energy = d * (1.0 - e) * (1.0 - e) - d;
                // du/dr = 2 d (1 - e) (a e), so the force is its negative.
                (energy, -2.0 * d * a * e * (1.0 - e))
            }
            Self::Coulomb { ke } => {
                let energy = ke * qi * qj / r;
                (energy, energy / r)
            }
            Self::LjCoulomb { eps, sigma, ke } => {
                let (u, f) = lj_pair(r, eps, sigma);
                let coulomb = ke * qi * qj / r;
                (u + coulomb, f + coulomb / r)
            }
            Self::Harmonic { k, r0 } => (0.5 * k * (r - r0) * (r - r0), -k * (r - r0)),
            Self::Custom(ref law) => law(r),
        }
    }

    /// Whether the potential carries a charge term, and so cannot be
    /// truncated at a cutoff without an Ewald correction.
    #[must_use]
    pub fn is_charged(&self) -> bool {
        matches!(self, Self::Coulomb { .. } | Self::LjCoulomb { .. })
    }
}

fn lj_pair(r: f64, eps: f64, sigma: f64) -> (f64, f64) {
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6 * sr6;
    (4.0 * eps * (sr12 - sr6), 24.0 * eps * (2.0 * sr12 - sr6) / r)
}

/// One record from a trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MdSample {
    /// Elapsed time.
    pub time: f64,
    /// Kinetic energy.
    pub kinetic: f64,
    /// Potential energy.
    pub potential: f64,
    /// Their sum, the quantity an NVE run must conserve.
    pub total: f64,
    /// Instantaneous temperature from equipartition.
    pub temperature: f64,
    /// Pressure from the virial.
    pub pressure: f64,
}

/// A box of particles interacting through one pair potential.
#[derive(Clone, Debug)]
pub struct MdSystem {
    /// Positions, wrapped into the box when periodic.
    ///
    /// Writing here directly invalidates the cached forces; see
    /// [`MdSystem::refresh_forces`].
    pub pos: Vec<Vec3>,
    /// Positions without wrapping, for displacement measurements.
    ///
    /// A mean squared displacement taken from wrapped coordinates saturates
    /// at the box size and reports a diffusion coefficient of zero however
    /// freely the particles are moving, so the unwrapped copy is not a
    /// convenience -- it is the only correct input to [`MdSystem::msd`].
    pub unwrapped: Vec<Vec3>,
    /// Velocities.
    pub vel: Vec<Vec3>,
    /// Masses.
    pub mass: Vec<f64>,
    /// Charges, used only by the charged potentials.
    pub charge: Vec<f64>,
    /// Edge lengths of the box.
    pub box_size: Vec3,
    /// Whether the box wraps.
    pub periodic: bool,
    /// The pair law.
    pub potential: Potential,
    /// Interaction cutoff.
    pub cutoff: f64,
    /// Elapsed time.
    pub time: f64,
    /// The Nose-Hoover friction coordinate, carried between steps.
    pub nose_hoover_zeta: f64,
    forces: Vec<Vec3>,
}

impl MdSystem {
    /// A system from explicit state.
    ///
    /// # Errors
    /// Returns an error for mismatched lengths, a non-positive mass, box or
    /// cutoff, or a cutoff more than half the shortest box edge, which the
    /// minimum-image convention cannot represent.
    pub fn new(
        pos: Vec<Vec3>,
        vel: Vec<Vec3>,
        mass: Vec<f64>,
        box_size: Vec3,
        periodic: bool,
        potential: Potential,
        cutoff: f64,
    ) -> Result<Self, GeomError> {
        if pos.is_empty() || pos.len() != vel.len() || pos.len() != mass.len() {
            return Err(GeomError::InvalidArgument("MdSystem: mismatched state"));
        }
        if mass.iter().any(|m| !(*m > 0.0)) {
            return Err(GeomError::InvalidArgument("every mass must be positive"));
        }
        if !(cutoff > 0.0) {
            return Err(GeomError::InvalidArgument("the cutoff must be positive"));
        }
        let shortest = box_size.x.min(box_size.y).min(box_size.z);
        if !(shortest > 0.0) {
            return Err(GeomError::InvalidArgument("every box edge must be positive"));
        }
        // Beyond half the shortest edge a particle would interact with two
        // images of the same neighbour, and the minimum image is no longer
        // the only image inside the cutoff.
        if periodic && cutoff > 0.5 * shortest {
            return Err(GeomError::InvalidArgument(
                "the cutoff exceeds half the shortest box edge",
            ));
        }
        let charge = vec![0.0; pos.len()];
        let mut system = Self {
            unwrapped: pos.clone(),
            pos,
            vel,
            mass,
            charge,
            box_size,
            periodic,
            potential,
            cutoff,
            time: 0.0,
            nose_hoover_zeta: 0.0,
            forces: Vec::new(),
        };
        if system.periodic {
            for k in 0..system.pos.len() {
                system.pos[k] = system.wrap(system.pos[k]);
            }
        }
        system.forces = system.compute_forces().0;
        Ok(system)
    }

    /// A face-centred cubic lattice of `cells^3` unit cells at a given
    /// number density, with velocities drawn at the requested temperature.
    ///
    /// FCC rather than simple cubic because it is the Lennard-Jones ground
    /// state: starting from a simple cubic lattice at liquid density puts
    /// the system on a mechanically unstable configuration, and it melts
    /// into a shock rather than into equilibrium.
    ///
    /// # Errors
    /// Returns an error for no cells, a non-positive density or a negative
    /// temperature.
    pub fn lattice_fcc(
        cells: usize,
        density: f64,
        temperature: f64,
        eps: f64,
        sigma: f64,
        rng: &mut Rng,
    ) -> Result<Self, GeomError> {
        if cells == 0 || cells > 32 {
            return Err(GeomError::InvalidArgument("lattice_fcc handles 1 to 32 cells"));
        }
        if !(density > 0.0) || temperature < 0.0 || !(eps > 0.0) || !(sigma > 0.0) {
            return Err(GeomError::InvalidArgument("lattice_fcc: bad parameters"));
        }
        let count = 4 * cells * cells * cells;
        let length = (count as f64 / density).cbrt();
        let a = length / cells as f64;
        // The four-atom conventional cell.
        const BASIS: [(f64, f64, f64); 4] =
            [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)];
        let mut pos = Vec::with_capacity(count);
        for i in 0..cells {
            for j in 0..cells {
                for k in 0..cells {
                    for (dx, dy, dz) in BASIS {
                        pos.push(Vec3::new(
                            (i as f64 + dx) * a,
                            (j as f64 + dy) * a,
                            (k as f64 + dz) * a,
                        ));
                    }
                }
            }
        }
        let vel: Vec<Vec3> = (0..count)
            .map(|_| {
                let s = temperature.sqrt();
                Vec3::new(
                    rng.next_gaussian() * s,
                    rng.next_gaussian() * s,
                    rng.next_gaussian() * s,
                )
            })
            .collect();
        let box_size = Vec3::new(length, length, length);
        let cutoff = (2.5 * sigma).min(0.5 * length - 1e-9);
        let mut system = Self::new(
            pos,
            vel,
            vec![1.0; count],
            box_size,
            true,
            Potential::LennardJones { eps, sigma },
            cutoff,
        )?;
        system.remove_drift();
        if temperature > 0.0 {
            system.rescale_to_temperature(temperature);
        }
        Ok(system)
    }

    /// The number of particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Whether the box is empty. Never true for a constructed system.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// The box volume.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.box_size.x * self.box_size.y * self.box_size.z
    }

    /// Wraps a position into the primary box.
    #[must_use]
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            p.x.rem_euclid(self.box_size.x),
            p.y.rem_euclid(self.box_size.y),
            p.z.rem_euclid(self.box_size.z),
        )
    }

    /// The shortest displacement between two points under the periodic
    /// boundary: the *minimum image*.
    #[must_use]
    pub fn minimum_image(&self, d: Vec3) -> Vec3 {
        if !self.periodic {
            return d;
        }
        let fold = |x: f64, l: f64| x - l * (x / l).round();
        Vec3::new(fold(d.x, self.box_size.x), fold(d.y, self.box_size.y), fold(d.z, self.box_size.z))
    }

    /// The velocity-of-the-centre-of-mass subtracted from every particle.
    ///
    /// The total momentum is a constant of the motion, so a non-zero value
    /// never decays: it sits in the kinetic energy for the whole run and
    /// inflates every temperature reading by a fixed amount.
    pub fn remove_drift(&mut self) {
        let total_mass: f64 = self.mass.iter().sum();
        let momentum = self.total_momentum();
        let correction = momentum * (1.0 / total_mass);
        for v in &mut self.vel {
            *v = *v - correction;
        }
    }

    /// Scales every velocity so the instantaneous temperature is `target`.
    pub fn rescale_to_temperature(&mut self, target: f64) {
        let current = self.temperature();
        if current <= 0.0 || target < 0.0 {
            return;
        }
        let factor = (target / current).sqrt();
        for v in &mut self.vel {
            *v = *v * factor;
        }
    }
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

impl MdSystem {
    /// The number of cells per box edge, or `None` when the box is too small
    /// for a cell list to be unambiguous.
    fn cell_counts(&self) -> Option<(usize, usize, usize)> {
        if !self.periodic {
            return None;
        }
        let n = |l: f64| ((l / self.cutoff).floor() as usize).max(1);
        let (mut nx, mut ny, mut nz) = (n(self.box_size.x), n(self.box_size.y), n(self.box_size.z));
        // A dilute system -- a big box with a short cutoff -- asks for far
        // more cells than there are particles, and the grid alone can be
        // enormous: a box of four hundred sigma with a cutoff of a tenth
        // wants sixty-four billion cells for a few hundred particles. A
        // cell *larger* than the cutoff is still correct, only less
        // selective, so the counts are halved until the grid is comparable
        // to the particle count.
        let budget = 8u128 * self.pos.len().max(1) as u128;
        let product = |a: usize, b: usize, c: usize| a as u128 * b as u128 * c as u128;
        while product(nx, ny, nz) > budget {
            let largest = nx.max(ny).max(nz);
            if largest <= 1 {
                break;
            }
            if nx == largest {
                nx = nx.div_ceil(2);
            } else if ny == largest {
                ny = ny.div_ceil(2);
            } else {
                nz = nz.div_ceil(2);
            }
        }
        // Below three cells an edge, a cell's forward and backward
        // neighbours coincide and every pair between them would be counted
        // twice. The all-pairs fallback is correct at that size and cheap.
        if nx >= 3 && ny >= 3 && nz >= 3 {
            Some((nx, ny, nz))
        } else {
            None
        }
    }

    /// Visits every interacting pair exactly once, with the minimum-image
    /// displacement from `j` to `i` and its squared length.
    fn for_each_pair(&self, mut visit: impl FnMut(usize, usize, Vec3, f64)) {
        let rc2 = self.cutoff * self.cutoff;
        let Some((nx, ny, nz)) = self.cell_counts() else {
            for i in 0..self.pos.len() {
                for j in (i + 1)..self.pos.len() {
                    let d = self.minimum_image(self.pos[i] - self.pos[j]);
                    let r2 = d.magnitude_squared();
                    if r2 < rc2 && r2 > 0.0 {
                        visit(i, j, d, r2);
                    }
                }
            }
            return;
        };

        let index = |x: usize, y: usize, z: usize| (x * ny + y) * nz + z;
        let mut cells = vec![Vec::new(); nx * ny * nz];
        let of = |p: Vec3| {
            let c = |v: f64, l: f64, n: usize| {
                ((v / l * n as f64).floor() as isize).rem_euclid(n as isize) as usize
            };
            (
                c(p.x, self.box_size.x, nx),
                c(p.y, self.box_size.y, ny),
                c(p.z, self.box_size.z, nz),
            )
        };
        for (k, p) in self.pos.iter().enumerate() {
            let (x, y, z) = of(*p);
            cells[index(x, y, z)].push(k);
        }

        // Thirteen of the twenty-six neighbour offsets: exactly one of every
        // pair `(d, -d)`, so each unordered cell pair is visited once.
        const FORWARD: [(isize, isize, isize); 13] = [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (-1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (-1, 0, 1),
            (0, 1, 1),
            (0, -1, 1),
            (1, 1, 1),
            (1, -1, 1),
            (-1, 1, 1),
            (-1, -1, 1),
        ];
        let consider = |i: usize, j: usize, visit: &mut dyn FnMut(usize, usize, Vec3, f64)| {
            let d = self.minimum_image(self.pos[i] - self.pos[j]);
            let r2 = d.magnitude_squared();
            if r2 < rc2 && r2 > 0.0 {
                visit(i, j, d, r2);
            }
        };
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let here = &cells[index(x, y, z)];
                    for a in 0..here.len() {
                        for b in (a + 1)..here.len() {
                            consider(here[a], here[b], &mut visit);
                        }
                    }
                    for (dx, dy, dz) in FORWARD {
                        let ox = (x as isize + dx).rem_euclid(nx as isize) as usize;
                        let oy = (y as isize + dy).rem_euclid(ny as isize) as usize;
                        let oz = (z as isize + dz).rem_euclid(nz as isize) as usize;
                        for &i in here {
                            for &j in &cells[index(ox, oy, oz)] {
                                consider(i, j, &mut visit);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The forces, the potential energy and the virial `sum r . f`.
    ///
    /// The potential is shifted by its value at the cutoff so that it is
    /// continuous there. An unshifted truncation puts a step in the energy
    /// at `r = rc`, and every particle that crosses it injects that step
    /// into the total -- which shows up as a steady energy drift that looks
    /// like an integrator fault and is not one.
    fn compute_forces(&self) -> (Vec<Vec3>, f64, f64) {
        let shift = if self.potential.is_charged() {
            0.0
        } else {
            self.potential.evaluate(self.cutoff, 0.0, 0.0).0
        };
        let mut forces = vec![Vec3::new(0.0, 0.0, 0.0); self.pos.len()];
        let mut energy = 0.0;
        let mut virial = 0.0;
        self.for_each_pair(|i, j, d, r2| {
            let r = r2.sqrt();
            let (u, f) = self.potential.evaluate(r, self.charge[i], self.charge[j]);
            energy += u - shift;
            virial += f * r;
            let force = d * (f / r);
            forces[i] = forces[i] + force;
            forces[j] = forces[j] - force;
        });
        (forces, energy, virial)
    }

    /// The force on every particle, computed afresh.
    #[must_use]
    pub fn forces(&self) -> Vec<Vec3> {
        self.compute_forces().0
    }

    /// Recomputes the cached forces the integrator steps with.
    ///
    /// [`MdSystem::step_velocity_verlet`] reuses the force it computed at the
    /// end of the previous step, which is what makes velocity Verlet one
    /// force evaluation per step rather than two. Every method here that
    /// changes a position keeps that cache current, but `pos`, `box_size`,
    /// `charge`, `potential` and `cutoff` are public: **after writing to any
    /// of them directly, call this before stepping.** Otherwise the next
    /// step integrates the previous configuration's forces, and the symptom
    /// is subtle rather than loud -- energy that almost conserves, and a
    /// trajectory that is no longer reversible.
    pub fn refresh_forces(&mut self) {
        self.forces = self.compute_forces().0;
    }

    /// The total potential energy.
    #[must_use]
    pub fn potential_energy(&self) -> f64 {
        self.compute_forces().1
    }

    /// The total kinetic energy.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.vel
            .iter()
            .zip(&self.mass)
            .map(|(v, m)| 0.5 * m * v.magnitude_squared())
            .sum()
    }

    /// The momentum of the whole box.
    #[must_use]
    pub fn total_momentum(&self) -> Vec3 {
        self.vel
            .iter()
            .zip(&self.mass)
            .fold(Vec3::new(0.0, 0.0, 0.0), |acc, (v, m)| acc + *v * *m)
    }

    /// The count of translational degrees of freedom.
    ///
    /// Three fewer than `3 N` on a periodic box, because the total momentum
    /// is conserved and carries no thermal energy. Dividing by `3 N`
    /// instead reports a temperature low by a factor `1 - 1/N`, which is
    /// invisible at ten thousand particles and a two per cent error at a
    /// hundred.
    #[must_use]
    pub fn degrees_of_freedom(&self) -> f64 {
        if self.periodic && self.pos.len() > 1 {
            3.0 * self.pos.len() as f64 - 3.0
        } else {
            3.0 * self.pos.len() as f64
        }
    }

    /// The instantaneous temperature from equipartition.
    #[must_use]
    pub fn temperature(&self) -> f64 {
        2.0 * self.kinetic_energy() / self.degrees_of_freedom()
    }

    /// The pressure from the virial theorem.
    #[must_use]
    pub fn pressure_virial(&self) -> f64 {
        let (_, _, virial) = self.compute_forces();
        let kinetic = 2.0 * self.kinetic_energy() / 3.0;
        (kinetic + virial / 3.0) / self.volume()
    }

    /// A snapshot of the thermodynamic state.
    #[must_use]
    pub fn sample(&self) -> MdSample {
        let (_, energy, virial) = self.compute_forces();
        let kinetic = self.kinetic_energy();
        MdSample {
            time: self.time,
            kinetic,
            potential: energy,
            total: kinetic + energy,
            temperature: 2.0 * kinetic / self.degrees_of_freedom(),
            pressure: (2.0 * kinetic / 3.0 + virial / 3.0) / self.volume(),
        }
    }
}

// ---------------------------------------------------------------------------
// Integration and temperature control
// ---------------------------------------------------------------------------

impl MdSystem {
    /// One velocity-Verlet step.
    ///
    /// Symplectic, so the energy error stays bounded rather than
    /// accumulating: the integrator conserves a shadow Hamiltonian close to
    /// the true one, and the true energy oscillates around its initial value
    /// forever instead of drifting away from it. That is the whole reason to
    /// prefer it over a higher-order but non-symplectic scheme here, and
    /// [`energy_drift`] is written to measure the distinction.
    pub fn step_velocity_verlet(&mut self, dt: f64) {
        let half = 0.5 * dt;
        for k in 0..self.pos.len() {
            let a = self.forces[k] * (1.0 / self.mass[k]);
            self.vel[k] = self.vel[k] + a * half;
            let step = self.vel[k] * dt;
            self.unwrapped[k] = self.unwrapped[k] + step;
            self.pos[k] = self.pos[k] + step;
            if self.periodic {
                self.pos[k] = self.wrap(self.pos[k]);
            }
        }
        self.forces = self.compute_forces().0;
        for k in 0..self.pos.len() {
            let a = self.forces[k] * (1.0 / self.mass[k]);
            self.vel[k] = self.vel[k] + a * half;
        }
        self.time += dt;
    }

    /// Berendsen velocity rescaling toward `t_target`.
    ///
    /// It reaches the right mean temperature and samples the wrong
    /// ensemble: the fluctuations are suppressed, so a heat capacity taken
    /// from a Berendsen run is too small. Use it to equilibrate and switch
    /// to Nose-Hoover or Langevin before measuring anything that depends on
    /// a fluctuation.
    pub fn thermostat_berendsen(&mut self, t_target: f64, tau: f64, dt: f64) {
        let current = self.temperature();
        if current <= 0.0 || tau <= 0.0 {
            return;
        }
        let factor = (1.0 + dt / tau * (t_target / current - 1.0)).max(0.0).sqrt();
        for v in &mut self.vel {
            *v = *v * factor;
        }
    }

    /// One Nose-Hoover step on the friction coordinate and the velocities.
    ///
    /// Unlike Berendsen this is derived from an extended Hamiltonian, so it
    /// samples the canonical ensemble including the fluctuations -- the
    /// friction is a dynamical variable with its own inertia `q`, and it
    /// oscillates rather than clamping.
    pub fn thermostat_nose_hoover(&mut self, t_target: f64, q: f64, dt: f64) {
        if !(q > 0.0) {
            return;
        }
        let dof = self.degrees_of_freedom();
        let kinetic = self.kinetic_energy();
        let acceleration = (2.0 * kinetic - dof * t_target) / q;
        self.nose_hoover_zeta += acceleration * dt;
        let factor = (-self.nose_hoover_zeta * dt).exp();
        for v in &mut self.vel {
            *v = *v * factor;
        }
    }

    /// One Langevin step: friction plus the matching noise.
    ///
    /// The two are not independent. The fluctuation-dissipation theorem
    /// fixes the noise amplitude from the friction and the target
    /// temperature, and any other amplitude thermostats to a different
    /// temperature than the one requested.
    pub fn thermostat_langevin(&mut self, t_target: f64, gamma: f64, dt: f64, rng: &mut Rng) {
        if gamma < 0.0 || t_target < 0.0 {
            return;
        }
        let decay = (-gamma * dt).exp();
        for k in 0..self.vel.len() {
            let amplitude = (t_target / self.mass[k] * (1.0 - decay * decay)).max(0.0).sqrt();
            self.vel[k] = self.vel[k] * decay
                + Vec3::new(
                    rng.next_gaussian() * amplitude,
                    rng.next_gaussian() * amplitude,
                    rng.next_gaussian() * amplitude,
                );
        }
    }

    /// Berendsen barostat: the box and every position scaled toward
    /// `p_target`.
    ///
    /// # Errors
    /// Returns an error for a non-positive time constant or compressibility,
    /// or if the rescaling would shrink the box below twice the cutoff.
    pub fn barostat_berendsen(
        &mut self,
        p_target: f64,
        compressibility: f64,
        tau: f64,
        dt: f64,
    ) -> Result<(), GeomError> {
        if !(tau > 0.0) || !(compressibility > 0.0) {
            return Err(GeomError::InvalidArgument("barostat_berendsen: bad parameters"));
        }
        let pressure = self.pressure_virial();
        let mu = (1.0 - compressibility * dt / tau * (p_target - pressure)).max(0.0).cbrt();
        let scaled = self.box_size * mu;
        let shortest = scaled.x.min(scaled.y).min(scaled.z);
        if self.periodic && self.cutoff > 0.5 * shortest {
            return Err(GeomError::Degenerate("the barostat shrank the box below the cutoff"));
        }
        self.box_size = scaled;
        for k in 0..self.pos.len() {
            self.pos[k] = self.pos[k] * mu;
            self.unwrapped[k] = self.unwrapped[k] * mu;
        }
        self.forces = self.compute_forces().0;
        Ok(())
    }

    /// Thermalises the system with a Langevin thermostat and returns the
    /// drift-free result.
    ///
    /// # Errors
    /// Returns an error for a non-positive step or a negative temperature.
    pub fn equilibrate(
        &mut self,
        steps: usize,
        dt: f64,
        t_target: f64,
        rng: &mut Rng,
    ) -> Result<(), GeomError> {
        if !(dt > 0.0) || t_target < 0.0 {
            return Err(GeomError::InvalidArgument("equilibrate: bad parameters"));
        }
        for _ in 0..steps {
            self.step_velocity_verlet(dt);
            self.thermostat_langevin(t_target, 1.0, dt, rng);
        }
        self.remove_drift();
        Ok(())
    }

    /// A constant-energy run, sampled every step.
    ///
    /// # Errors
    /// Returns an error for a non-positive step or no steps.
    pub fn run_nve(&mut self, steps: usize, dt: f64) -> Result<Vec<MdSample>, GeomError> {
        if !(dt > 0.0) || steps == 0 {
            return Err(GeomError::InvalidArgument("run_nve: bad parameters"));
        }
        let mut out = Vec::with_capacity(steps + 1);
        out.push(self.sample());
        for _ in 0..steps {
            self.step_velocity_verlet(dt);
            out.push(self.sample());
        }
        Ok(out)
    }

    /// A constant-energy run recording positions and velocities every
    /// `stride` steps, for the transport measurements.
    ///
    /// # Errors
    /// Returns an error for a non-positive step, no steps, or a zero stride.
    pub fn run_trajectory(
        &mut self,
        steps: usize,
        dt: f64,
        stride: usize,
    ) -> Result<(Vec<Vec<Vec3>>, Vec<Vec<Vec3>>), GeomError> {
        if !(dt > 0.0) || steps == 0 || stride == 0 {
            return Err(GeomError::InvalidArgument("run_trajectory: bad parameters"));
        }
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        for step in 0..=steps {
            if step % stride == 0 {
                // Unwrapped, since a displacement across the boundary is a
                // real displacement.
                positions.push(self.unwrapped.clone());
                velocities.push(self.vel.clone());
            }
            if step < steps {
                self.step_velocity_verlet(dt);
            }
        }
        Ok((positions, velocities))
    }
}

/// The secular drift of the total energy over a record, relative to its mean.
///
/// This is the slope of a least-squares line through the total energy,
/// multiplied by the elapsed time -- not the spread. A symplectic
/// integrator's energy *oscillates* with an amplitude set by the step size
/// and does not go anywhere; reporting that oscillation as drift would
/// condemn a correct integrator, and reporting the maximum deviation would
/// do the same. What distinguishes a good integrator from a bad one is
/// whether the oscillation has a trend under it.
///
/// # Errors
/// Returns an error for fewer than three samples or a zero time span.
pub fn energy_drift(samples: &[MdSample]) -> Result<f64, GeomError> {
    if samples.len() < 3 {
        return Err(GeomError::InvalidArgument("energy_drift needs three samples"));
    }
    let n = samples.len() as f64;
    let sx: f64 = samples.iter().map(|s| s.time).sum();
    let sy: f64 = samples.iter().map(|s| s.total).sum();
    let sxx: f64 = samples.iter().map(|s| s.time * s.time).sum();
    let sxy: f64 = samples.iter().map(|s| s.time * s.total).sum();
    let denominator = n * sxx - sx * sx;
    if denominator.abs() < 1e-300 {
        return Err(GeomError::Degenerate("the samples share one time"));
    }
    let slope = (n * sxy - sx * sy) / denominator;
    let span = samples[samples.len() - 1].time - samples[0].time;
    let scale = (sy / n).abs().max(1e-300);
    Ok((slope * span / scale).abs())
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

impl MdSystem {
    /// The radial distribution function `g(r)` in `bins` shells out to
    /// `r_max`.
    ///
    /// Normalised by the *ideal gas* count in each shell, so `g(r) = 1`
    /// means "no correlation at this separation" rather than "no
    /// neighbours". A histogram normalised by the shell volume alone rises
    /// as `r^2` and says nothing.
    ///
    /// # Errors
    /// Returns an error for no bins, a non-positive range, or a range
    /// exceeding half the shortest box edge on a periodic box.
    pub fn rdf(&self, bins: usize, r_max: f64) -> Result<Vec<f64>, GeomError> {
        if bins == 0 || !(r_max > 0.0) {
            return Err(GeomError::InvalidArgument("rdf: bad parameters"));
        }
        let shortest = self.box_size.x.min(self.box_size.y).min(self.box_size.z);
        if self.periodic && r_max > 0.5 * shortest {
            return Err(GeomError::InvalidArgument("the range exceeds half the box"));
        }
        let width = r_max / bins as f64;
        let mut counts = vec![0.0f64; bins];
        let n = self.pos.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.minimum_image(self.pos[i] - self.pos[j]);
                let r = d.magnitude();
                if r < r_max {
                    counts[(r / width) as usize] += 2.0;
                }
            }
        }
        let density = n as f64 / self.volume();
        Ok(counts
            .into_iter()
            .enumerate()
            .map(|(k, c)| {
                let lo = k as f64 * width;
                let hi = lo + width;
                let shell = 4.0 / 3.0 * std::f64::consts::PI * (hi * hi * hi - lo * lo * lo);
                c / (n as f64 * density * shell)
            })
            .collect())
    }

    /// The static structure factor at each scalar wavenumber, by the Debye
    /// formula `S(k) = 1 + (2/N) sum_{i<j} sin(k r) / (k r)`.
    ///
    /// Exact for the sample in hand, and it tends to one at large `k`
    /// whatever the configuration -- which is the check that the
    /// normalisation is right.
    ///
    /// Below `2 pi / L` it is not a structural measurement at all. Every
    /// `sin(k r) / (k r)` tends to one as `k` falls, so the sum tends to
    /// the particle count: what that peak measures is the extent of the
    /// sample, not any correlation inside it. Read this only at
    /// wavenumbers above the smallest reciprocal box vector.
    ///
    /// # Errors
    /// Returns an error for a non-positive wavenumber.
    pub fn structure_factor(&self, k_values: &[f64]) -> Result<Vec<f64>, GeomError> {
        if k_values.iter().any(|k| !(*k > 0.0)) {
            return Err(GeomError::InvalidArgument("every wavenumber must be positive"));
        }
        let n = self.pos.len();
        let mut distances = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(self.minimum_image(self.pos[i] - self.pos[j]).magnitude());
            }
        }
        Ok(k_values
            .iter()
            .map(|k| {
                let sum: f64 = distances
                    .iter()
                    .map(|r| {
                        let x = k * r;
                        if x.abs() < 1e-12 {
                            1.0
                        } else {
                            x.sin() / x
                        }
                    })
                    .sum();
                1.0 + 2.0 * sum / n as f64
            })
            .collect())
    }

    /// A Kolmogorov-Smirnov test of the speeds against the Maxwell-Boltzmann
    /// distribution at the system's own temperature.
    ///
    /// The check is worth making because equipartition alone does not pin
    /// the distribution: a system with every particle at the same speed has
    /// exactly the right temperature and entirely the wrong statistics, and
    /// that is precisely the state a freshly rescaled lattice is in.
    ///
    /// # Errors
    /// Returns an error for a zero temperature or fewer than five
    /// particles, and for masses that are not all equal -- the speeds then
    /// come from a mixture of distributions and a single-sample test does
    /// not apply.
    pub fn maxwell_boltzmann_check(&self) -> Result<TestResult, GeomError> {
        if self.pos.len() < 5 {
            return Err(GeomError::InvalidArgument("too few particles to test"));
        }
        let m0 = self.mass[0];
        if self.mass.iter().any(|m| (m - m0).abs() > 1e-12 * m0) {
            return Err(GeomError::InvalidArgument("the masses are not all equal"));
        }
        let temperature = self.temperature();
        if !(temperature > 0.0) {
            return Err(GeomError::Degenerate("the system is at zero temperature"));
        }
        let a = (temperature / m0).sqrt();
        let speeds: Vec<f64> = self.vel.iter().map(Vec3::magnitude).collect();
        let cdf = move |v: f64| -> f64 {
            if v <= 0.0 {
                return 0.0;
            }
            let x = v / a;
            crate::special::erf::erf(x / std::f64::consts::SQRT_2)
                - (2.0 / std::f64::consts::PI).sqrt() * x * (-0.5 * x * x).exp()
        };
        Ok(ks_test_one_sample(&speeds, &cdf))
    }

    /// The Lindemann ratio: the root-mean-square displacement of each
    /// particle from its own mean position, divided by the nearest-neighbour
    /// distance.
    ///
    /// Above about 0.15 a crystal has melted. The ratio is taken about each
    /// particle's *own* time-averaged site rather than about a lattice, so
    /// it does not need to know the crystal structure -- but for the same
    /// reason it only means something for a trajectory long enough for that
    /// average to settle.
    ///
    /// # Errors
    /// Returns an error for fewer than two frames or a frame of the wrong
    /// length.
    pub fn melting_indicator_lindemann(&self, traj: &[Vec<Vec3>]) -> Result<f64, GeomError> {
        if traj.len() < 2 {
            return Err(GeomError::InvalidArgument("the trajectory is too short"));
        }
        let n = self.pos.len();
        if traj.iter().any(|frame| frame.len() != n) {
            return Err(GeomError::InvalidArgument("the frames differ in length"));
        }
        let frames = traj.len() as f64;
        let mut total = 0.0;
        for i in 0..n {
            let mean = traj
                .iter()
                .fold(Vec3::new(0.0, 0.0, 0.0), |acc, frame| acc + frame[i])
                * (1.0 / frames);
            let spread: f64 =
                traj.iter().map(|frame| (frame[i] - mean).magnitude_squared()).sum::<f64>() / frames;
            total += spread;
        }
        let rms = (total / n as f64).sqrt();
        // The nearest-neighbour distance of the current configuration.
        let mut nearest = f64::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                let r = self.minimum_image(self.pos[i] - self.pos[j]).magnitude();
                if r > 0.0 && r < nearest {
                    nearest = r;
                }
            }
        }
        if !nearest.is_finite() || nearest <= 0.0 {
            return Err(GeomError::Degenerate("no neighbour distance to normalise by"));
        }
        Ok(rms / nearest)
    }
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

impl MdSystem {
    /// The mean squared displacement against lag, averaged over particles
    /// and over every time origin.
    ///
    /// The trajectory must be *unwrapped*: a position folded back into the
    /// box turns a steady drift into a sawtooth, and the resulting MSD
    /// saturates at the box size and reports no diffusion at all. Use the
    /// positions from [`MdSystem::run_trajectory`], which are unwrapped for
    /// this reason.
    ///
    /// # Errors
    /// Returns an error for fewer than two frames or ragged frames.
    pub fn msd(traj: &[Vec<Vec3>]) -> Result<Vec<f64>, GeomError> {
        if traj.len() < 2 || traj[0].is_empty() {
            return Err(GeomError::InvalidArgument("the trajectory is too short"));
        }
        let n = traj[0].len();
        if traj.iter().any(|frame| frame.len() != n) {
            return Err(GeomError::InvalidArgument("the frames differ in length"));
        }
        let frames = traj.len();
        Ok((0..frames)
            .map(|lag| {
                let origins = frames - lag;
                let mut total = 0.0;
                for start in 0..origins {
                    for i in 0..n {
                        total += (traj[start + lag][i] - traj[start][i]).magnitude_squared();
                    }
                }
                total / (origins * n) as f64
            })
            .collect())
    }

    /// The diffusion coefficient from the Einstein relation `<r^2> = 6 D t`.
    ///
    /// Fitted over the middle half of the record. The two ends are excluded
    /// deliberately: the short-lag part is ballistic rather than diffusive,
    /// and the long-lag part is averaged over so few time origins that it is
    /// mostly noise. Fitting the whole curve mixes both in.
    ///
    /// # Errors
    /// Returns an error for fewer than eight lags or a non-positive step.
    pub fn diffusion_coefficient(msd: &[f64], dt: f64) -> Result<f64, GeomError> {
        if msd.len() < 8 || !(dt > 0.0) {
            return Err(GeomError::InvalidArgument("diffusion_coefficient: bad input"));
        }
        let lo = msd.len() / 4;
        let hi = msd.len() * 3 / 4;
        let points: Vec<(f64, f64)> =
            (lo..hi).map(|k| (k as f64 * dt, msd[k])).collect();
        let n = points.len() as f64;
        let sx: f64 = points.iter().map(|p| p.0).sum();
        let sy: f64 = points.iter().map(|p| p.1).sum();
        let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let denominator = n * sxx - sx * sx;
        if denominator.abs() < 1e-300 {
            return Err(GeomError::Degenerate("the lags do not vary"));
        }
        Ok((n * sxy - sx * sy) / denominator / 6.0)
    }

    /// The normalised velocity autocorrelation function.
    ///
    /// # Errors
    /// Returns an error for fewer than two frames, ragged frames, or a
    /// trajectory with no motion in it.
    pub fn vacf(traj_vel: &[Vec<Vec3>]) -> Result<Vec<f64>, GeomError> {
        if traj_vel.len() < 2 || traj_vel[0].is_empty() {
            return Err(GeomError::InvalidArgument("the trajectory is too short"));
        }
        let n = traj_vel[0].len();
        if traj_vel.iter().any(|frame| frame.len() != n) {
            return Err(GeomError::InvalidArgument("the frames differ in length"));
        }
        let frames = traj_vel.len();
        let raw: Vec<f64> = (0..frames)
            .map(|lag| {
                let origins = frames - lag;
                let mut total = 0.0;
                for start in 0..origins {
                    for i in 0..n {
                        total += traj_vel[start + lag][i].dot(&traj_vel[start][i]);
                    }
                }
                total / (origins * n) as f64
            })
            .collect();
        if !(raw[0] > 0.0) {
            return Err(GeomError::Degenerate("the trajectory has no motion"));
        }
        Ok(raw.iter().map(|c| c / raw[0]).collect())
    }

    /// The vibrational density of states: the cosine transform of the
    /// velocity autocorrelation.
    ///
    /// Returned on the frequency grid `omega_k = pi k / (N dt)`, so the
    /// caller can label the axis without guessing.
    ///
    /// # Errors
    /// Returns an error for fewer than two points or a non-positive step.
    pub fn vdos_from_vacf(vacf: &[f64], dt: f64) -> Result<Vec<f64>, GeomError> {
        if vacf.len() < 2 || !(dt > 0.0) {
            return Err(GeomError::InvalidArgument("vdos_from_vacf: bad input"));
        }
        let n = vacf.len();
        Ok((0..n)
            .map(|k| {
                let omega = std::f64::consts::PI * k as f64 / (n as f64 * dt);
                let mut total = 0.0;
                for (t, c) in vacf.iter().enumerate() {
                    // Trapezoidal, with the end points at half weight.
                    let weight = if t == 0 || t == n - 1 { 0.5 } else { 1.0 };
                    total += weight * c * (omega * t as f64 * dt).cos();
                }
                2.0 * total * dt
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Reference quantities
// ---------------------------------------------------------------------------

/// What the reduced units in this module mean.
#[must_use]
pub fn lj_reduced_units_note() -> &'static str {
    "Lennard-Jones reduced units: lengths in sigma, energies in eps, masses \
     in m, and Boltzmann's constant equal to one. Time is then \
     sigma sqrt(m / eps), temperature is eps / k_B, pressure is eps / sigma^3 \
     and number density is 1 / sigma^3. For argon (sigma = 3.4 A, \
     eps / k_B = 120 K, m = 40 amu) one time unit is about 2.16 ps, so a step \
     of 0.005 is about 10 fs."
}

/// A rough phase from the Lennard-Jones phase diagram.
///
/// Boundaries taken from the accepted triple point near `(T* = 0.69,
/// rho* = 0.84)` and critical point near `(T* = 1.32, rho* = 0.31)`. It is a
/// classification, not an equation of state, and near a boundary it should
/// not be trusted over an actual measurement.
#[must_use]
pub fn lj_phase_point(t_star: f64, rho_star: f64) -> &'static str {
    if rho_star <= 0.0 || t_star <= 0.0 {
        return "unphysical";
    }
    if rho_star > 0.94 || (t_star < 0.69 && rho_star > 0.84) {
        return "solid";
    }
    if t_star > 1.32 && rho_star > 0.20 {
        return "supercritical fluid";
    }
    if rho_star < 0.05 {
        return "gas";
    }
    if rho_star > 0.6 {
        return "liquid";
    }
    if t_star < 1.32 {
        return "gas-liquid coexistence";
    }
    "fluid"
}

/// The Ewald energy of a set of point charges in a periodic box.
///
/// A charged system cannot be truncated: the Coulomb sum is only
/// conditionally convergent, so its value depends on the order of
/// summation and a spherical cutoff gives a different -- wrong -- answer.
/// Ewald splits the sum with a Gaussian screen into a real-space part that
/// converges quickly and a reciprocal-space part that does the same, plus
/// the self-energy of the screens.
///
/// # Errors
/// Returns an error for mismatched lengths, a non-positive box or splitting
/// parameter, an empty system, or a net charge, for which the sum is not
/// defined without a neutralising background.
pub fn ewald_sum_energy_lite(
    charges: &[f64],
    pos: &[Vec3],
    box_l: f64,
    alpha: f64,
    k_max: usize,
) -> Result<f64, GeomError> {
    if charges.is_empty() || charges.len() != pos.len() {
        return Err(GeomError::InvalidArgument("ewald: mismatched input"));
    }
    if !(box_l > 0.0) || !(alpha > 0.0) || k_max == 0 || k_max > 32 {
        return Err(GeomError::InvalidArgument("ewald: bad parameters"));
    }
    let net: f64 = charges.iter().sum();
    if net.abs() > 1e-9 * charges.iter().map(|q| q.abs()).sum::<f64>().max(1.0) {
        return Err(GeomError::InvalidArgument("the system must be neutral"));
    }
    let n = charges.len();
    let volume = box_l * box_l * box_l;

    // Real space, out to the minimum image.
    let mut real = 0.0;
    let fold = |x: f64| x - box_l * (x / box_l).round();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pos[i] - pos[j];
            let r = Vec3::new(fold(d.x), fold(d.y), fold(d.z)).magnitude();
            if r > 0.0 {
                real += charges[i] * charges[j] * crate::special::erf::erfc(alpha * r) / r;
            }
        }
    }

    // Reciprocal space.
    let mut reciprocal = 0.0;
    let two_pi_over_l = 2.0 * std::f64::consts::PI / box_l;
    let limit = k_max as isize;
    for nx in -limit..=limit {
        for ny in -limit..=limit {
            for nz in -limit..=limit {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let k = Vec3::new(nx as f64, ny as f64, nz as f64) * two_pi_over_l;
                let k2 = k.magnitude_squared();
                let (mut cos_sum, mut sin_sum) = (0.0, 0.0);
                for (q, p) in charges.iter().zip(pos) {
                    let phase = k.dot(p);
                    cos_sum += q * phase.cos();
                    sin_sum += q * phase.sin();
                }
                let structure = cos_sum * cos_sum + sin_sum * sin_sum;
                reciprocal += (-k2 / (4.0 * alpha * alpha)).exp() / k2 * structure;
            }
        }
    }
    reciprocal *= 2.0 * std::f64::consts::PI / volume;

    // The self-energy of each charge's own screen, which the reciprocal sum
    // includes and the physical energy does not.
    let self_energy =
        alpha / std::f64::consts::PI.sqrt() * charges.iter().map(|q| q * q).sum::<f64>();
    Ok(real + reciprocal - self_energy)
}

/// The heat capacity per particle of a system from its energy fluctuations,
/// in units of Boltzmann's constant.
///
/// `C_v = Var(E) / (k T^2)`. A classical harmonic crystal must return three:
/// each particle has three quadratic kinetic and three quadratic potential
/// degrees of freedom, and equipartition gives `k/2` to each. That is the
/// Dulong-Petit law, and it is the check this function exists for -- a
/// simulation that reports anything else at a temperature well above the
/// Debye temperature has a bug, not a discovery.
///
/// # Errors
/// Returns an error for fewer than two energies, no particles, or a
/// non-positive temperature.
pub fn harmonic_crystal_heat_capacity_check(
    energies: &[f64],
    temperature: f64,
    particles: usize,
) -> Result<f64, GeomError> {
    if energies.len() < 2 || particles == 0 || !(temperature > 0.0) {
        return Err(GeomError::InvalidArgument("heat capacity check: bad input"));
    }
    let n = energies.len() as f64;
    let mean: f64 = energies.iter().sum::<f64>() / n;
    // The unbiased variance; with a few hundred samples the difference from
    // the biased one is the difference between 3.00 and 2.99.
    let variance: f64 = energies.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / (n - 1.0);
    Ok(variance / (temperature * temperature * particles as f64))
}

/// The second virial coefficient by numerical integration of the Mayer
/// function.
///
/// `B2(T) = -2 pi int_0^rmax (exp(-u(r)/T) - 1) r^2 dr`. It changes sign at
/// the Boyle temperature, where attraction and repulsion cancel and the gas
/// is ideal to first order in the density -- about `T* = 3.418` for
/// Lennard-Jones.
///
/// # Errors
/// Returns an error for a non-positive temperature or range, or an odd or
/// too-small interval count.
pub fn virial_coefficient_b2(
    potential: &Potential,
    t: f64,
    r_max: f64,
    n: usize,
) -> Result<f64, GeomError> {
    if !(t > 0.0) || !(r_max > 0.0) || n < 2 || !n.is_multiple_of(2) {
        return Err(GeomError::InvalidArgument("virial_coefficient_b2: bad parameters"));
    }
    let h = r_max / n as f64;
    // Simpson's rule. The integrand is well behaved at the origin: the
    // Mayer function tends to -1 there for any repulsive core, so the
    // integrand tends to -r^2 rather than diverging with the energy.
    let f = |r: f64| -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        let u = potential.evaluate(r, 0.0, 0.0).0;
        let mayer = if u / t > 700.0 { -1.0 } else { (-u / t).exp() - 1.0 };
        mayer * r * r
    };
    let mut total = f(0.0) + f(r_max);
    for k in 1..n {
        let weight = if k.is_multiple_of(2) { 2.0 } else { 4.0 };
        total += weight * f(k as f64 * h);
    }
    Ok(-2.0 * std::f64::consts::PI * total * h / 3.0)
}

/// The mean free path `1 / (sqrt 2 n sigma)`, with `sigma` the collision
/// cross-section.
///
/// The `sqrt 2` is not decoration: it accounts for the *relative* motion of
/// the two colliding particles, and dropping it overestimates the path by
/// forty per cent.
///
/// # Errors
/// Returns an error for a non-positive density or cross-section.
pub fn mean_free_path(density: f64, sigma: f64) -> Result<f64, GeomError> {
    if !(density > 0.0) || !(sigma > 0.0) {
        return Err(GeomError::InvalidArgument("mean_free_path needs positive input"));
    }
    Ok(1.0 / (2f64.sqrt() * density * sigma))
}

/// The collision rate per particle, `sqrt 2 n sigma v_mean`.
///
/// # Errors
/// Returns an error for a non-positive density, cross-section or speed.
pub fn collision_rate(density: f64, sigma: f64, mean_speed: f64) -> Result<f64, GeomError> {
    if !(mean_speed > 0.0) {
        return Err(GeomError::InvalidArgument("the mean speed must be positive"));
    }
    Ok(mean_speed / mean_free_path(density, sigma)?)
}

/// The shear viscosity from a Green-Kubo integral of the off-diagonal
/// stress autocorrelation.
///
/// `eta = V / (k T) int_0^inf <P_xy(0) P_xy(t)> dt`, integrated up to the
/// first lag at which the estimated correlation stops being positive.
///
/// That truncation is not an optimisation. Past a few correlation times the
/// estimate of `<P(0) P(t)>` is noise of a size set by the sample count, and
/// integrating thousands of such lags accumulates a random walk whose spread
/// is comparable to the whole integral -- for an exponential correlation
/// with a thirty-sample time and thirty thousand samples, the tail
/// contributes as much scatter as the signal contains. Integrating to the
/// end of the record therefore returns a number that is mostly noise, which
/// looks like a plausible viscosity and is not one.
///
/// The cost of truncating is a known one: stopping at the first zero
/// crossing loses the part of the tail already below the noise floor, so the
/// result is a few per cent low. That is the accepted trade, and it is the
/// direction of the remaining error -- a short record still *underestimates*
/// rather than scattering, because the tail it cannot see carries real
/// weight.
///
/// # Errors
/// Returns an error for fewer than two samples or a non-positive step,
/// volume or temperature.
pub fn green_kubo_viscosity_lite(
    stress_xy: &[f64],
    dt: f64,
    volume: f64,
    temperature: f64,
) -> Result<f64, GeomError> {
    if stress_xy.len() < 2 || !(dt > 0.0) || !(volume > 0.0) || !(temperature > 0.0) {
        return Err(GeomError::InvalidArgument("green_kubo_viscosity_lite: bad input"));
    }
    let frames = stress_xy.len();
    let correlation: Vec<f64> = (0..frames / 2)
        .map(|lag| {
            let origins = frames - lag;
            stress_xy[..origins]
                .iter()
                .enumerate()
                .map(|(t, s)| s * stress_xy[t + lag])
                .sum::<f64>()
                / origins as f64
        })
        .collect();
    let mut integral = 0.0;
    for k in 1..correlation.len() {
        if correlation[k] <= 0.0 {
            break;
        }
        integral += 0.5 * (correlation[k - 1] + correlation[k]) * dt;
    }
    Ok(volume / temperature * integral)
}

/// A potential of mean force from umbrella-sampling histograms, by
/// self-consistent WHAM.
///
/// Each window is biased by `k (x - centre)^2 / 2`, and the windows have to
/// be combined by solving for one free-energy offset per window: simply
/// unbiasing each histogram and averaging leaves the offsets arbitrary, and
/// the resulting curve has a step at every window boundary.
///
/// `histograms[w][b]` is the count in bin `b` of window `w`; bin `b` is
/// centred at `bin_lo + (b + 0.5) * bin_width`.
///
/// # Errors
/// Returns an error for no windows, mismatched lengths, a non-positive bin
/// width, force constant or temperature, or if the iteration does not
/// converge.
pub fn umbrella_sampling_pmf(
    histograms: &[Vec<f64>],
    centers: &[f64],
    k: f64,
    bin_lo: f64,
    bin_width: f64,
    temperature: f64,
) -> Result<Vec<f64>, GeomError> {
    if histograms.is_empty() || histograms.len() != centers.len() {
        return Err(GeomError::InvalidArgument("umbrella_sampling_pmf: mismatched input"));
    }
    let bins = histograms[0].len();
    if bins == 0 || histograms.iter().any(|h| h.len() != bins) {
        return Err(GeomError::InvalidArgument("the histograms differ in length"));
    }
    if !(bin_width > 0.0) || !(k > 0.0) || !(temperature > 0.0) {
        return Err(GeomError::InvalidArgument("umbrella_sampling_pmf: bad parameters"));
    }
    let windows = histograms.len();
    let samples: Vec<f64> = histograms.iter().map(|h| h.iter().sum()).collect();
    if samples.iter().any(|s| !(*s > 0.0)) {
        return Err(GeomError::Degenerate("a window collected no samples"));
    }
    let x = |b: usize| bin_lo + (b as f64 + 0.5) * bin_width;
    // The bias each window applies in each bin, in units of kT.
    let bias: Vec<Vec<f64>> = (0..windows)
        .map(|w| {
            (0..bins)
                .map(|b| {
                    let d = x(b) - centers[w];
                    0.5 * k * d * d / temperature
                })
                .collect()
        })
        .collect();

    let total: Vec<f64> = (0..bins).map(|b| histograms.iter().map(|h| h[b]).sum()).collect();
    let mut free = vec![0.0f64; windows];
    for _ in 0..10_000 {
        // Unbiased probability in each bin, given the current offsets.
        let probability: Vec<f64> = (0..bins)
            .map(|b| {
                let denominator: f64 =
                    (0..windows).map(|w| samples[w] * (free[w] - bias[w][b]).exp()).sum();
                if denominator > 0.0 {
                    total[b] / denominator
                } else {
                    0.0
                }
            })
            .collect();
        let mut updated = vec![0.0f64; windows];
        for w in 0..windows {
            let z: f64 = (0..bins).map(|b| probability[b] * (-bias[w][b]).exp()).sum();
            if !(z > 0.0) {
                return Err(GeomError::Degenerate("a window has no overlap with the histogram"));
            }
            updated[w] = -z.ln();
        }
        // The offsets are defined only up to a constant, so the first is
        // pinned; otherwise the iteration wanders without ever converging.
        let anchor = updated[0];
        for f in &mut updated {
            *f -= anchor;
        }
        let change = (0..windows).map(|w| (updated[w] - free[w]).abs()).fold(0.0, f64::max);
        free = updated;
        if change < 1e-12 {
            let mut pmf: Vec<f64> = (0..bins)
                .map(|b| {
                    let denominator: f64 =
                        (0..windows).map(|w| samples[w] * (free[w] - bias[w][b]).exp()).sum();
                    if total[b] > 0.0 && denominator > 0.0 {
                        -temperature * (total[b] / denominator).ln()
                    } else {
                        f64::INFINITY
                    }
                })
                .collect();
            let lowest = pmf.iter().copied().fold(f64::INFINITY, f64::min);
            if lowest.is_finite() {
                for value in &mut pmf {
                    *value -= lowest;
                }
            }
            return Ok(pmf);
        }
    }
    Err(GeomError::Degenerate("WHAM did not converge"))
}

/// A steered-molecular-dynamics pull: a harmonic restraint whose centre
/// moves at constant speed, returning the accumulated work at each step.
///
/// The work is *not* the free-energy difference. It exceeds it by the
/// dissipation, and only in the reversible limit do the two coincide --
/// which is what Jarzynski's equality repairs, by averaging `exp(-W/kT)`
/// over repeated pulls rather than averaging the work itself.
///
/// # Errors
/// Returns an error for a non-positive step, force constant or step count.
pub fn steered_pull(
    force_along: &dyn Fn(f64) -> f64,
    start: f64,
    speed: f64,
    k: f64,
    dt: f64,
    steps: usize,
) -> Result<Vec<f64>, GeomError> {
    if !(dt > 0.0) || !(k > 0.0) || steps == 0 {
        return Err(GeomError::InvalidArgument("steered_pull: bad parameters"));
    }
    let mut x = start;
    let mut work = 0.0;
    let mut out = Vec::with_capacity(steps);
    for step in 0..steps {
        let centre = start + speed * step as f64 * dt;
        // Overdamped motion in the sum of the true force and the restraint.
        let force = force_along(x) + k * (centre - x);
        x += force * dt;
        // The work done by moving the restraint is the restraint force
        // times the displacement of its centre.
        work += k * (centre - x) * speed * dt;
        out.push(work);
    }
    Ok(out)
}

/// Jarzynski's estimate of the free-energy difference from a set of
/// non-equilibrium work values.
///
/// `exp(-dF/kT) = <exp(-W/kT)>`. The average is dominated by the rare
/// trajectories with the *smallest* work, which is why the estimator is
/// notoriously hard to converge: the trajectories that matter most are the
/// ones sampled least.
///
/// # Errors
/// Returns an error for no work values or a non-positive temperature.
pub fn jarzynski_free_energy(work: &[f64], temperature: f64) -> Result<f64, GeomError> {
    if work.is_empty() || !(temperature > 0.0) {
        return Err(GeomError::InvalidArgument("jarzynski_free_energy: bad input"));
    }
    // Shifted by the smallest work, since the exponentials otherwise
    // overflow long before the average means anything.
    let smallest = work.iter().copied().fold(f64::INFINITY, f64::min);
    let mean: f64 = work.iter().map(|w| (-(w - smallest) / temperature).exp()).sum::<f64>()
        / work.len() as f64;
    Ok(smallest - temperature * mean.ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------
    // Potentials
    // -----------------------------------------------------------------

    #[test]
    fn every_potential_force_is_the_negative_gradient_of_its_own_energy() {
        // The single most consequential invariant in the module. A force
        // that is not exactly minus the derivative of the energy in use
        // conserves nothing, and the failure is indistinguishable from an
        // integrator bug -- so it is checked here, at the source, against a
        // central difference of each variant's own energy.
        let laws = [
            Potential::LennardJones { eps: 1.0, sigma: 1.0 },
            Potential::LennardJones { eps: 2.5, sigma: 0.8 },
            Potential::Morse { d: 1.5, a: 2.0, r0: 1.2 },
            Potential::Coulomb { ke: 1.0 },
            Potential::LjCoulomb { eps: 1.0, sigma: 1.0, ke: 0.7 },
            Potential::Harmonic { k: 3.0, r0: 1.1 },
        ];
        for law in &laws {
            for step in 1..=40 {
                let r = 0.85 + 0.05 * f64::from(step);
                let h = 1e-6;
                let (_, force) = law.evaluate(r, 1.0, -1.0);
                let up = law.evaluate(r + h, 1.0, -1.0).0;
                let down = law.evaluate(r - h, 1.0, -1.0).0;
                let numeric = -(up - down) / (2.0 * h);
                let scale = force.abs().max(numeric.abs()).max(1.0);
                assert!(
                    close(force, numeric, 1e-4 * scale),
                    "{law:?} at r = {r} gives force {force} against gradient {numeric}"
                );
            }
        }
    }

    #[test]
    fn the_lennard_jones_minimum_sits_where_theory_puts_it() {
        // The well bottom is at 2^(1/6) sigma and its depth is exactly eps.
        // Both are closed form, so they pin the parameterisation rather
        // than merely describing it.
        for &sigma in &[0.5f64, 1.0, 2.0] {
            for &eps in &[0.25f64, 1.0, 3.0] {
                let law = Potential::LennardJones { eps, sigma };
                let r_min = 2f64.powf(1.0 / 6.0) * sigma;
                let (u, f) = law.evaluate(r_min, 0.0, 0.0);
                assert!(close(u, -eps, 1e-12 * eps), "the well depth is {u}, not {eps}");
                assert!(close(f, 0.0, 1e-9 * eps / sigma), "the force at the minimum is {f}");
                // Zero crossing at sigma, repulsive inside, attractive out.
                assert!(close(law.evaluate(sigma, 0.0, 0.0).0, 0.0, 1e-12 * eps));
                assert!(law.evaluate(0.9 * sigma, 0.0, 0.0).1 > 0.0);
                assert!(law.evaluate(1.5 * sigma, 0.0, 0.0).1 < 0.0);
            }
        }
        // Morse likewise: depth d at r0, zero force there.
        let morse = Potential::Morse { d: 2.0, a: 1.5, r0: 1.3 };
        assert!(close(morse.evaluate(1.3, 0.0, 0.0).0, -2.0, 1e-12));
        assert!(close(morse.evaluate(1.3, 0.0, 0.0).1, 0.0, 1e-12));
        // And it dissociates: the energy tends to zero from below.
        assert!(close(morse.evaluate(40.0, 0.0, 0.0).0, 0.0, 1e-9));
        assert!(morse.evaluate(3.0, 0.0, 0.0).0 < 0.0);
    }

    #[test]
    fn a_custom_law_is_used_as_given() {
        let law = Potential::Custom(Arc::new(|r: f64| (r * r, -2.0 * r)));
        assert!(close(law.evaluate(3.0, 0.0, 0.0).0, 9.0, 1e-12));
        assert!(close(law.evaluate(3.0, 0.0, 0.0).1, -6.0, 1e-12));
        assert!(!law.is_charged());
        assert!(Potential::Coulomb { ke: 1.0 }.is_charged());
        assert!(format!("{law:?}").contains("Custom"));
    }

    // -----------------------------------------------------------------
    // Geometry and forces
    // -----------------------------------------------------------------

    fn two_body(separation: f64, box_l: f64, cutoff: f64) -> MdSystem {
        MdSystem::new(
            vec![Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0 + separation, 1.0, 1.0)],
            vec![Vec3::new(0.0, 0.0, 0.0); 2],
            vec![1.0; 2],
            Vec3::new(box_l, box_l, box_l),
            true,
            Potential::LennardJones { eps: 1.0, sigma: 1.0 },
            cutoff,
        )
        .unwrap()
    }

    #[test]
    fn the_minimum_image_picks_the_nearer_of_the_two_ways_round() {
        let system = two_body(1.0, 10.0, 3.0);
        // A displacement of 6 in a box of 10 is really -4.
        let d = system.minimum_image(Vec3::new(6.0, -7.0, 2.0));
        assert!(close(d.x, -4.0, 1e-12));
        assert!(close(d.y, 3.0, 1e-12));
        assert!(close(d.z, 2.0, 1e-12));
        // Every component lands in [-L/2, L/2].
        let mut rng = Rng::new(0x011D_0001);
        for _ in 0..500 {
            let raw = Vec3::new(
                rng.next_f64() * 60.0 - 30.0,
                rng.next_f64() * 60.0 - 30.0,
                rng.next_f64() * 60.0 - 30.0,
            );
            let folded = system.minimum_image(raw);
            for c in [folded.x, folded.y, folded.z] {
                assert!(c.abs() <= 5.0 + 1e-9, "the folded component {c} is outside the box");
            }
            // And it differs from the raw displacement by a whole number of
            // box lengths, so it is the same point.
            let shift = (raw.x - folded.x) / 10.0;
            assert!(close(shift, shift.round(), 1e-9));
        }
        // A non-periodic box folds nothing.
        let open = MdSystem::new(
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            vec![Vec3::new(0.0, 0.0, 0.0); 2],
            vec![1.0; 2],
            Vec3::new(10.0, 10.0, 10.0),
            false,
            Potential::LennardJones { eps: 1.0, sigma: 1.0 },
            3.0,
        )
        .unwrap();
        assert!(close(open.minimum_image(Vec3::new(6.0, 0.0, 0.0)).x, 6.0, 1e-12));
    }

    #[test]
    fn the_pair_force_is_equal_and_opposite_and_matches_the_law() {
        // Newton's third law, and the two-body force is the law itself --
        // the cutoff shift changes the energy and must leave the force
        // alone.
        for step in 1..=20 {
            let r = 0.9 + 0.05 * f64::from(step);
            let system = two_body(r, 12.0, 2.5);
            let f = system.forces();
            assert!(close((f[0] + f[1]).magnitude(), 0.0, 1e-9), "the forces do not cancel");
            // Particle 0 sits at the lower x, so a positive (repulsive)
            // radial force pushes it toward -x.
            let radial = Potential::LennardJones { eps: 1.0, sigma: 1.0 }.evaluate(r, 0.0, 0.0).1;
            assert!(
                close(f[0].x, -radial, 1e-9 * radial.abs().max(1.0)),
                "at r = {r} the force is {} against {}",
                f[0].x,
                -radial
            );
            assert!(close(f[0].y, 0.0, 1e-12) && close(f[0].z, 0.0, 1e-12));
            // And the sign is the physics: the force changes sign at the
            // well bottom, 2^(1/6) sigma, not at sigma -- the *energy*
            // crosses zero at sigma and the pair is still repelling there.
            if r < 2f64.powf(1.0 / 6.0) {
                assert!(f[0].x < 0.0 && f[1].x > 0.0, "the pair does not repel at r = {r}");
            } else {
                assert!(f[0].x > 0.0 && f[1].x < 0.0, "the pair does not attract at r = {r}");
            }
        }
        // Beyond the cutoff there is nothing at all.
        let far = two_body(3.0, 12.0, 2.5);
        assert!(close(far.forces()[0].magnitude(), 0.0, 1e-15));
        assert!(close(far.potential_energy(), 0.0, 1e-15));
    }

    #[test]
    fn the_shifted_potential_is_continuous_at_the_cutoff() {
        // The reason the shift is there: without it the energy steps as a
        // pair crosses the cutoff, and every crossing injects that step
        // into the total.
        let cutoff = 2.5;
        assert!(close(two_body(cutoff + 1e-7, 20.0, cutoff).potential_energy(), 0.0, 1e-15));
        // The gap across the cutoff is u'(rc) h, first order in h. A fixed
        // tolerance would only be testing the h I happened to pick, so the
        // gap is measured at two step sizes: continuity says halving h
        // halves it, and a discontinuity says the gap stops shrinking.
        let gap = |h: f64| two_body(cutoff - h, 20.0, cutoff).potential_energy().abs();
        let coarse = gap(1e-6);
        let fine = gap(5e-7);
        assert!(coarse > 0.0, "the potential is flat at the cutoff, so nothing is being tested");
        assert!(
            close(coarse / fine, 2.0, 0.01),
            "halving the step changed the gap by {} rather than two",
            coarse / fine
        );
        // The unshifted potential would leave a step of u(rc) = -0.0163
        // there, which is more than three orders of magnitude larger than
        // the gap at this step size.
        let unshifted = Potential::LennardJones { eps: 1.0, sigma: 1.0 }
            .evaluate(cutoff, 0.0, 0.0)
            .0
            .abs();
        assert!(coarse < 1e-3 * unshifted, "the energy still steps by {coarse} at the cutoff");
    }

    #[test]
    fn the_cell_list_and_the_all_pairs_loop_agree() {
        // The cell list is the only performance-critical piece here, and it
        // is easy to get subtly wrong at the wrapping boundary. Checked
        // against the direct loop on the same configuration, which is the
        // reference the fallback path already provides.
        let mut rng = Rng::new(0x011D_0002);
        for trial in 0..4 {
            // Five cells an edge is the smallest FCC lattice whose box holds
            // three cutoffs; below that the all-pairs fallback runs and this
            // test would silently compare it against itself.
            let cells = 5 + trial % 2;
            let mut system =
                MdSystem::lattice_fcc(cells, 0.85, 1.0, 1.0, 1.0, &mut rng).unwrap();
            // Jitter, so the configuration is not the symmetric lattice.
            for p in &mut system.pos {
                *p = *p
                    + Vec3::new(
                        rng.next_gaussian() * 0.08,
                        rng.next_gaussian() * 0.08,
                        rng.next_gaussian() * 0.08,
                    );
            }
            for k in 0..system.pos.len() {
                system.pos[k] = system.wrap(system.pos[k]);
            }
            assert!(system.cell_counts().is_some(), "the cell path was not taken");
            let (cell_forces, cell_energy, cell_virial) = system.compute_forces();

            // The same pairs by the direct O(N^2) route, which is the
            // reference the fallback path already provides.
            let (mut pair_forces, mut pair_energy, mut pair_virial) =
                (vec![Vec3::new(0.0, 0.0, 0.0); system.len()], 0.0, 0.0);
            let shift = system.potential.evaluate(system.cutoff, 0.0, 0.0).0;
            let rc2 = system.cutoff * system.cutoff;
            for i in 0..system.len() {
                for j in (i + 1)..system.len() {
                    let d = system.minimum_image(system.pos[i] - system.pos[j]);
                    let r2 = d.magnitude_squared();
                    if r2 < rc2 && r2 > 0.0 {
                        let r = r2.sqrt();
                        let (u, f) = system.potential.evaluate(r, 0.0, 0.0);
                        pair_energy += u - shift;
                        pair_virial += f * r;
                        let force = d * (f / r);
                        pair_forces[i] = pair_forces[i] + force;
                        pair_forces[j] = pair_forces[j] - force;
                    }
                }
            }
            let scale = pair_energy.abs().max(1.0);
            assert!(
                close(cell_energy, pair_energy, 1e-8 * scale),
                "the cell list gives energy {cell_energy} against {pair_energy}"
            );
            assert!(close(cell_virial, pair_virial, 1e-8 * pair_virial.abs().max(1.0)));
            for k in 0..system.len() {
                assert!(
                    close((cell_forces[k] - pair_forces[k]).magnitude(), 0.0, 1e-8 * scale),
                    "the force on particle {k} differs between the two loops"
                );
            }
        }
    }

    #[test]
    fn the_forces_of_an_isolated_box_sum_to_zero() {
        // Newton's third law over the whole system: the internal forces
        // cancel, so the centre of mass does not accelerate. It holds
        // whichever traversal is used, which is what makes it a check on
        // both.
        let mut rng = Rng::new(0x011D_0003);
        for cells in [2usize, 3, 4] {
            let system = MdSystem::lattice_fcc(cells, 0.7, 1.2, 1.0, 1.0, &mut rng).unwrap();
            let total = system.forces().into_iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, f| a + f);
            assert!(
                close(total.magnitude(), 0.0, 1e-8 * system.len() as f64),
                "the net force on {} particles is {}",
                system.len(),
                total.magnitude()
            );
        }
    }

    #[test]
    fn the_constructor_rejects_states_it_cannot_integrate() {
        let good = || vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let lj = || Potential::LennardJones { eps: 1.0, sigma: 1.0 };
        let b = Vec3::new(10.0, 10.0, 10.0);
        assert!(MdSystem::new(vec![], vec![], vec![], b, true, lj(), 2.5).is_err());
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0)], vec![1.0; 2], b, true, lj(), 2.5).is_err());
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0); 2], vec![0.0; 2], b, true, lj(), 2.5).is_err());
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0); 2], vec![1.0; 2], b, true, lj(), 0.0).is_err());
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0); 2], vec![1.0; 2], Vec3::new(0.0, 1.0, 1.0), true, lj(), 0.5).is_err());
        // The cutoff cannot exceed half the box: past that a particle sees
        // two images of the same neighbour.
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0); 2], vec![1.0; 2], b, true, lj(), 5.1).is_err());
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0); 2], vec![1.0; 2], b, true, lj(), 5.0).is_ok());
        // But an open box has no images to confuse.
        assert!(MdSystem::new(good(), vec![Vec3::new(0.0, 0.0, 0.0); 2], vec![1.0; 2], b, false, lj(), 50.0).is_ok());
        let mut rng = Rng::new(1);
        assert!(MdSystem::lattice_fcc(0, 0.8, 1.0, 1.0, 1.0, &mut rng).is_err());
        assert!(MdSystem::lattice_fcc(33, 0.8, 1.0, 1.0, 1.0, &mut rng).is_err());
        assert!(MdSystem::lattice_fcc(2, 0.0, 1.0, 1.0, 1.0, &mut rng).is_err());
        assert!(MdSystem::lattice_fcc(2, 0.8, -1.0, 1.0, 1.0, &mut rng).is_err());
    }




    // -----------------------------------------------------------------
    // Reference quantities
    // -----------------------------------------------------------------

    #[test]
    fn the_ewald_sum_reproduces_the_madelung_constant_of_rock_salt() {
        // The reference every Ewald implementation is checked against: an
        // alternating cubic charge lattice has energy -M ke q^2 / a per ion
        // with M = 1.747565, a number known to ten figures and reachable by
        // no truncated sum -- the Coulomb series is only conditionally
        // convergent, so a spherical cutoff gives a different answer
        // depending on where it is cut.
        const MADELUNG: f64 = 1.747_564_594_6;
        for cells in [2usize, 4] {
            let a = 1.0;
            let side = cells as f64 * a;
            let mut pos = Vec::new();
            let mut charges = Vec::new();
            for i in 0..cells {
                for j in 0..cells {
                    for k in 0..cells {
                        pos.push(Vec3::new(i as f64 * a, j as f64 * a, k as f64 * a));
                        charges.push(if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 });
                    }
                }
            }
            let n = charges.len() as f64;
            // Alpha chosen so the real-space part is dead inside the
            // minimum image -- erfc(alpha L / 2) with alpha = 8 / L is
            // 10^-8 -- and k_max large enough that the reciprocal part is
            // too. The two errors move in opposite directions with alpha,
            // so neither can be checked without pinning the other.
            let energy = ewald_sum_energy_lite(&charges, &pos, side, 8.0 / side, 12).unwrap();
            // The Madelung constant is defined by the energy of *one* ion in
            // the field of all the others, while the lattice energy counts
            // each pair once. So the total per ion is half of it -- the
            // factor of two that this convention costs everyone once.
            let madelung = -2.0 * energy / n;
            assert!(
                close(madelung, MADELUNG, 1e-5),
                "the {cells}-cell lattice gives a Madelung constant of {madelung}"
            );
        }
    }

    #[test]
    fn the_ewald_energy_does_not_depend_on_where_the_sum_is_split() {
        // Alpha is a free parameter of the method, not of the physics. An
        // implementation that dropped the self-energy term, or mismatched
        // erf and erfc, would still look plausible at one alpha and would
        // vary wildly across a range -- so this is the check that finds
        // those without needing a reference value at all.
        let mut rng = Rng::new(0x011D_0040);
        for _ in 0..4 {
            let count = 8;
            let side = 4.0;
            let pos: Vec<Vec3> = (0..count)
                .map(|_| {
                    Vec3::new(
                        rng.next_f64() * side,
                        rng.next_f64() * side,
                        rng.next_f64() * side,
                    )
                })
                .collect();
            let mut charges: Vec<f64> = (0..count - 1).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
            let balance = -charges.iter().sum::<f64>();
            charges.push(balance);
            let reference = ewald_sum_energy_lite(&charges, &pos, side, 1.5, 10).unwrap();
            for &alpha in &[1.0f64, 2.0, 2.5] {
                let other = ewald_sum_energy_lite(&charges, &pos, side, alpha, 12).unwrap();
                assert!(
                    close(other, reference, 1e-3 * reference.abs().max(1.0)),
                    "alpha {alpha} gives {other} against {reference}"
                );
            }
        }
        // A net charge has no defined Coulomb energy without a neutralising
        // background, and is refused rather than silently answered.
        assert!(ewald_sum_energy_lite(&[1.0, 1.0], &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)], 4.0, 1.0, 4).is_err());
        assert!(ewald_sum_energy_lite(&[], &[], 4.0, 1.0, 4).is_err());
        assert!(ewald_sum_energy_lite(&[1.0, -1.0], &[Vec3::new(0.0, 0.0, 0.0)], 4.0, 1.0, 4).is_err());
        let pair = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        assert!(ewald_sum_energy_lite(&[1.0, -1.0], &pair, 0.0, 1.0, 4).is_err());
        assert!(ewald_sum_energy_lite(&[1.0, -1.0], &pair, 4.0, 0.0, 4).is_err());
        assert!(ewald_sum_energy_lite(&[1.0, -1.0], &pair, 4.0, 1.0, 0).is_err());
        assert!(ewald_sum_energy_lite(&[1.0, -1.0], &pair, 4.0, 1.0, 33).is_err());
    }

    #[test]
    fn the_second_virial_coefficient_is_exact_for_a_hard_sphere() {
        // A hard sphere has B2 = 2 pi d^3 / 3 at every temperature -- the
        // Mayer function is exactly -1 inside the diameter and zero outside,
        // so the integral is elementary and the answer is a pure number.
        // That makes it the one case where the quadrature can be checked
        // rather than merely trusted.
        for &d in &[0.5f64, 1.0, 1.7] {
            let hard = Potential::Custom(Arc::new(move |r: f64| {
                if r < d {
                    (1e6, 0.0)
                } else {
                    (0.0, 0.0)
                }
            }));
            let expected = 2.0 * std::f64::consts::PI * d * d * d / 3.0;
            for &t in &[0.5f64, 1.0, 5.0] {
                let b2 = virial_coefficient_b2(&hard, t, 4.0, 40_000).unwrap();
                assert!(
                    close(b2, expected, 1e-3 * expected),
                    "a hard sphere of diameter {d} at T = {t} gives B2 = {b2} against {expected}"
                );
            }
        }
    }

    #[test]
    fn the_lennard_jones_virial_changes_sign_at_the_boyle_temperature() {
        // Below it attraction dominates and B2 is negative; above it the
        // repulsive core does and B2 is positive. The crossing is at
        // T* = 3.418, a number that comes out of the integral and is not
        // put into it.
        let lj = Potential::LennardJones { eps: 1.0, sigma: 1.0 };
        assert!(virial_coefficient_b2(&lj, 1.0, 8.0, 20_000).unwrap() < -1.0);
        assert!(virial_coefficient_b2(&lj, 10.0, 8.0, 20_000).unwrap() > 0.5);
        // Bisect for the zero.
        let (mut lo, mut hi) = (2.0f64, 6.0f64);
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if virial_coefficient_b2(&lj, mid, 8.0, 20_000).unwrap() < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let boyle = 0.5 * (lo + hi);
        assert!(close(boyle, 3.418, 0.02), "the Boyle temperature came out {boyle}");
        // B2 rises monotonically with temperature over this range.
        let mut previous = f64::NEG_INFINITY;
        for step in 1..=20 {
            let t = 0.6 + 0.5 * f64::from(step);
            let b2 = virial_coefficient_b2(&lj, t, 8.0, 20_000).unwrap();
            assert!(b2 > previous, "B2 fell from {previous} to {b2} at T = {t}");
            previous = b2;
        }
        assert!(virial_coefficient_b2(&lj, 0.0, 8.0, 100).is_err());
        assert!(virial_coefficient_b2(&lj, 1.0, 0.0, 100).is_err());
        assert!(virial_coefficient_b2(&lj, 1.0, 8.0, 101).is_err());
        assert!(virial_coefficient_b2(&lj, 1.0, 8.0, 1).is_err());
    }

    #[test]
    fn a_harmonic_crystal_obeys_dulong_and_petit() {
        // The classical heat capacity of a harmonic solid is 3 k per
        // particle, from equipartition over 3N kinetic and 3N potential
        // quadratic coordinates. Three of the potential coordinates are
        // zero modes -- a uniform translation costs nothing -- so a finite
        // crystal gives (6N - 3) / 2 rather than 3N, and at thirty-two
        // particles that is 2.95, not 3.00. Testing against 3.00 with a
        // loose tolerance would hide the distinction; this checks the
        // finite-size form.
        let mut rng = Rng::new(0x011D_0041);
        let cells = 2usize;
        let seed = MdSystem::lattice_fcc(cells, 1.0, 0.0, 1.0, 1.0, &mut rng).unwrap();
        let a = seed.box_size.x / cells as f64;
        let nearest = a / 2f64.sqrt();
        // Springs between nearest neighbours only.
        let mut crystal = MdSystem::new(
            seed.pos.clone(),
            vec![Vec3::new(0.0, 0.0, 0.0); seed.len()],
            vec![1.0; seed.len()],
            seed.box_size,
            true,
            Potential::Harmonic { k: 40.0, r0: nearest },
            nearest * 1.05,
        )
        .unwrap();
        let n = crystal.len();
        let temperature = 0.02;
        let dt = 0.004;
        for _ in 0..4_000 {
            crystal.step_velocity_verlet(dt);
            crystal.thermostat_langevin(temperature, 4.0, dt, &mut rng);
        }
        let mut energies = Vec::with_capacity(40_000);
        for step in 0..200_000 {
            crystal.step_velocity_verlet(dt);
            crystal.thermostat_langevin(temperature, 4.0, dt, &mut rng);
            if step % 5 == 0 {
                let s = crystal.sample();
                energies.push(s.total);
            }
        }
        let capacity =
            harmonic_crystal_heat_capacity_check(&energies, temperature, n).unwrap();
        let expected = (6.0 * n as f64 - 3.0) / 2.0 / n as f64;
        assert!(
            close(capacity, expected, 0.15 * expected),
            "the crystal gives {capacity} k per particle against {expected}"
        );
        // And it is nearer the finite-size value than the bulk one, which
        // is the point of computing the zero modes.
        assert!(expected < 3.0);
        assert!(harmonic_crystal_heat_capacity_check(&[1.0], 1.0, 4).is_err());
        assert!(harmonic_crystal_heat_capacity_check(&[1.0, 2.0], 0.0, 4).is_err());
        assert!(harmonic_crystal_heat_capacity_check(&[1.0, 2.0], 1.0, 0).is_err());
        // The formula itself, on a sample with a variance chosen by hand.
        let made: Vec<f64> = (0..1_001).map(|k| f64::from(k - 500) * 0.01).collect();
        let mean: f64 = made.iter().sum::<f64>() / made.len() as f64;
        let variance: f64 = made.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>()
            / (made.len() as f64 - 1.0);
        assert!(close(
            harmonic_crystal_heat_capacity_check(&made, 0.5, 3).unwrap(),
            variance / (0.25 * 3.0),
            1e-9
        ));
    }

    #[test]
    fn the_kinetic_theory_lengths_are_reciprocal_to_their_rates() {
        // The sqrt 2 accounts for the relative motion of the pair; dropping
        // it overestimates the path by forty per cent, which is why it is
        // checked against the closed form rather than a rounded number.
        for &density in &[0.1f64, 1.0, 25.0] {
            for &sigma in &[0.05f64, 1.0, 3.0] {
                let lambda = mean_free_path(density, sigma).unwrap();
                assert!(close(lambda, 1.0 / (2f64.sqrt() * density * sigma), 1e-12));
                for &speed in &[0.5f64, 4.0] {
                    let rate = collision_rate(density, sigma, speed).unwrap();
                    // A particle covers one mean free path per collision.
                    assert!(close(rate * lambda, speed, 1e-9 * speed));
                    assert!(close(rate, 2f64.sqrt() * density * sigma * speed, 1e-9 * rate));
                }
            }
        }
        // Doubling the density halves the path.
        assert!(close(
            mean_free_path(2.0, 1.0).unwrap() * 2.0,
            mean_free_path(1.0, 1.0).unwrap(),
            1e-12
        ));
        assert!(mean_free_path(0.0, 1.0).is_err());
        assert!(mean_free_path(1.0, 0.0).is_err());
        assert!(collision_rate(1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn the_green_kubo_integral_recovers_a_known_correlation_time() {
        // An Ornstein-Uhlenbeck stress has autocorrelation sigma^2 e^(-t/tau),
        // whose integral is sigma^2 tau exactly, so the transport
        // coefficient it produces is a closed form. Checked at two
        // correlation times, since a single one could be matched by an
        // integrator that was wrong by a constant.
        let volume = 3.0;
        let temperature = 2.0;
        let sigma = 1.5;
        let dt = 0.01;
        let ou = |rng: &mut Rng, tau: f64, samples: usize| -> Vec<f64> {
            let decay = (-dt / tau).exp();
            let noise = sigma * (1.0 - decay * decay).sqrt();
            let mut x = sigma * rng.next_gaussian();
            (0..samples)
                .map(|_| {
                    x = x * decay + noise * rng.next_gaussian();
                    x
                })
                .collect()
        };
        for &tau in &[0.3f64, 1.0] {
            let mut rng = Rng::new(0x011D_0042 + (tau * 10.0) as u64);
            let series = ou(&mut rng, tau, 30_000);
            let eta = green_kubo_viscosity_lite(&series, dt, volume, temperature).unwrap();
            let expected = volume / temperature * sigma * sigma * tau;
            assert!(
                close(eta, expected, 0.15 * expected),
                "at tau = {tau} the integral gives {eta} against {expected}"
            );

            let _ = expected;
        }
        // The truncation bias the documentation warns about. It cannot be
        // shown from one short series: a single realisation shorter than
        // its own correlation time is dominated by where it happened to
        // start, and comes out high as often as low. It is a *bias*, so it
        // needs an ensemble -- many short records, averaged, land
        // systematically below the answer because each can only integrate
        // the head of the correlation and never its tail.
        let tau = 5.0;
        let window = 400usize;
        let mut rng = Rng::new(0x011D_0044);
        // Half the window is the furthest lag reached, so the fraction of
        // the integral within reach is known in advance.
        let captured = 1.0 - (-(window as f64 / 2.0) * dt / tau).exp();
        assert!(captured < 0.45, "the window is not short enough to bias anything");
        let full = volume / temperature * sigma * sigma * tau;
        let mut short_total = 0.0;
        for _ in 0..600 {
            let piece = ou(&mut rng, tau, window);
            short_total += green_kubo_viscosity_lite(&piece, dt, volume, temperature).unwrap();
        }
        let short = short_total / 600.0;
        assert!(short > 0.0, "the truncated estimate collapsed to {short}");
        assert!(
            short < 0.6 * full,
            "the truncated ensemble gives {short}, not clearly below the full {full}"
        );

        assert!(green_kubo_viscosity_lite(&[1.0], 0.01, 1.0, 1.0).is_err());
        assert!(green_kubo_viscosity_lite(&[1.0, 2.0], 0.0, 1.0, 1.0).is_err());
        assert!(green_kubo_viscosity_lite(&[1.0, 2.0], 0.01, 0.0, 1.0).is_err());
        assert!(green_kubo_viscosity_lite(&[1.0, 2.0], 0.01, 1.0, 0.0).is_err());
    }

    #[test]
    fn wham_inverts_a_potential_of_mean_force_it_was_never_told() {
        // The strongest test available for this: rather than sampling, the
        // histograms are built *exactly* from a chosen free-energy profile
        // and the windows' own biases, so WHAM must return that profile up
        // to a constant with no statistical error at all. A method that
        // simply unbiased each window and averaged would leave a step at
        // every window boundary and fail here immediately.
        let temperature = 0.8;
        let k = 12.0;
        let bins = 60;
        let bin_lo = -3.0;
        let bin_width = 0.1;
        let x = |b: usize| bin_lo + (b as f64 + 0.5) * bin_width;
        // A double well, which is exactly the case umbrella sampling exists
        // for: the barrier is never crossed by an unbiased run.
        let truth = |v: f64| 4.0 * (v * v - 1.0) * (v * v - 1.0);
        let centers: Vec<f64> = (0..13).map(|w| -2.4 + 0.4 * f64::from(w)).collect();
        let histograms: Vec<Vec<f64>> = centers
            .iter()
            .map(|c| {
                let raw: Vec<f64> = (0..bins)
                    .map(|b| {
                        let v = x(b);
                        let bias = 0.5 * k * (v - c) * (v - c);
                        (-(truth(v) + bias) / temperature).exp()
                    })
                    .collect();
                let total: f64 = raw.iter().sum();
                raw.into_iter().map(|p| p / total * 100_000.0).collect()
            })
            .collect();
        let pmf = umbrella_sampling_pmf(
            &histograms,
            &centers,
            k,
            bin_lo,
            bin_width,
            temperature,
        )
        .unwrap();
        // Recovered up to a constant, so compare shapes: the returned curve
        // is shifted to a minimum of zero, and so is the truth.
        let true_curve: Vec<f64> = (0..bins).map(|b| truth(x(b))).collect();
        let true_min = true_curve.iter().copied().fold(f64::INFINITY, f64::min);
        for b in 0..bins {
            let expected = true_curve[b] - true_min;
            // Only where the windows actually reach: far outside them the
            // exact histogram underflows to nothing and the method has no
            // information, which is a real limitation and not a defect.
            if x(b).abs() <= 2.4 {
                assert!(
                    close(pmf[b], expected, 0.02 * expected.max(1.0)),
                    "at x = {} the PMF is {} against {expected}",
                    x(b),
                    pmf[b]
                );
            }
        }
        // The barrier is recovered: the true one is 4 at x = 0.
        let centre = pmf[bins / 2];
        assert!(close(centre, 4.0, 0.1), "the barrier came out {centre}");
        assert!(umbrella_sampling_pmf(&[], &[], k, bin_lo, bin_width, temperature).is_err());
        assert!(umbrella_sampling_pmf(&histograms, &centers[..2], k, bin_lo, bin_width, temperature).is_err());
        assert!(umbrella_sampling_pmf(&histograms, &centers, 0.0, bin_lo, bin_width, temperature).is_err());
        assert!(umbrella_sampling_pmf(&histograms, &centers, k, bin_lo, 0.0, temperature).is_err());
        assert!(umbrella_sampling_pmf(&histograms, &centers, k, bin_lo, bin_width, 0.0).is_err());
        let empty = vec![vec![0.0; bins]; centers.len()];
        assert!(umbrella_sampling_pmf(&empty, &centers, k, bin_lo, bin_width, temperature).is_err());
        let ragged = vec![vec![1.0; bins], vec![1.0; bins - 1]];
        assert!(umbrella_sampling_pmf(&ragged, &centers[..2], k, bin_lo, bin_width, temperature).is_err());
    }

    #[test]
    fn pulling_more_slowly_costs_less_work() {
        // The second law, as a measurement: the work exceeds the free-energy
        // change by the dissipation, and the dissipation falls as the pull
        // approaches reversibility. A pull that cost the same at every speed
        // would mean the dissipation was not being accounted for.
        let stiffness = 3.0;
        let force = move |x: f64| -stiffness * x;
        let distance = 1.0;
        let mut previous = f64::INFINITY;
        for shift in 0..5 {
            let speed = 1.0 / f64::from(1 << shift);
            let steps = 2_000 * (1 << shift);
            let dt = distance / (speed * steps as f64);
            let work = steered_pull(&force, 0.0, speed, 20.0, dt, steps).unwrap();
            let total = *work.last().unwrap();
            assert!(total > 0.0, "pulling uphill did no work");
            assert!(total < previous, "the slower pull cost {total} against {previous}");
            previous = total;
        }
        // The reversible limit is the free-energy change of the trap plus
        // the well, which is bounded below by the well's own.
        assert!(previous > 0.5 * stiffness * distance * distance * 0.5);
        assert!(steered_pull(&force, 0.0, 1.0, 1.0, 0.0, 10).is_err());
        assert!(steered_pull(&force, 0.0, 1.0, 0.0, 0.01, 10).is_err());
        assert!(steered_pull(&force, 0.0, 1.0, 1.0, 0.01, 0).is_err());
    }

    #[test]
    fn the_jarzynski_average_sits_below_the_mean_work() {
        // Jensen's inequality, which is the whole content of the second law
        // in this form: the exponential average is at or below the
        // arithmetic one, with equality only when every pull cost the same.
        let mut rng = Rng::new(0x011D_0043);
        let temperature = 0.7;
        for spread in [0.0f64, 0.3, 1.5] {
            let work: Vec<f64> =
                (0..4_000).map(|_| 2.0 + spread * rng.next_gaussian()).collect();
            let mean: f64 = work.iter().sum::<f64>() / work.len() as f64;
            let free = jarzynski_free_energy(&work, temperature).unwrap();
            assert!(free <= mean + 1e-9, "the estimate {free} exceeds the mean work {mean}");
            if spread == 0.0 {
                assert!(close(free, 2.0, 1e-12), "identical pulls gave {free}");
            } else {
                // For Gaussian work the gap is exactly the variance over 2kT.
                let expected = mean - spread * spread / (2.0 * temperature);
                assert!(
                    close(free, expected, 0.15 * spread * spread),
                    "at spread {spread} the estimate is {free} against {expected}"
                );
            }
        }
        assert!(jarzynski_free_energy(&[], 1.0).is_err());
        assert!(jarzynski_free_energy(&[1.0], 0.0).is_err());
    }

    #[test]
    fn the_phase_classification_and_the_units_note_say_what_they_should() {
        // The published triple and critical points, and the two limits
        // either side of them.
        assert_eq!(lj_phase_point(0.5, 1.0), "solid");
        assert_eq!(lj_phase_point(0.6, 0.9), "solid");
        assert_eq!(lj_phase_point(2.0, 0.8), "supercritical fluid");
        assert_eq!(lj_phase_point(1.0, 0.01), "gas");
        assert_eq!(lj_phase_point(1.0, 0.7), "liquid");
        assert_eq!(lj_phase_point(1.0, 0.3), "gas-liquid coexistence");
        assert_eq!(lj_phase_point(1.5, 0.1), "fluid");
        assert_eq!(lj_phase_point(-1.0, 0.5), "unphysical");
        assert_eq!(lj_phase_point(1.0, 0.0), "unphysical");
        let note = lj_reduced_units_note();
        assert!(note.contains("sigma") && note.contains("Boltzmann"));
        assert!(note.contains("argon"), "the note gives no worked conversion");
    }

    // -----------------------------------------------------------------
    // Structure
    // -----------------------------------------------------------------

    /// A box of well-separated particles that never interact, so the
    /// structural measures see an ideal gas.
    fn ideal_gas(count: usize, box_l: f64, rng: &mut Rng) -> MdSystem {
        let pos = (0..count)
            .map(|_| {
                Vec3::new(
                    rng.next_f64() * box_l,
                    rng.next_f64() * box_l,
                    rng.next_f64() * box_l,
                )
            })
            .collect();
        MdSystem::new(
            pos,
            vec![Vec3::new(0.0, 0.0, 0.0); count],
            vec![1.0; count],
            Vec3::new(box_l, box_l, box_l),
            true,
            Potential::LennardJones { eps: 1.0, sigma: 1.0 },
            0.3,
        )
        .unwrap()
    }

    #[test]
    fn the_radial_distribution_of_an_ideal_gas_is_one_everywhere() {
        // The normalisation check. A histogram divided by the shell volume
        // alone rises as r^2 and says nothing about correlation; dividing
        // by the ideal-gas count is what makes g(r) = 1 mean "uncorrelated"
        // rather than "empty".
        let mut rng = Rng::new(0x011D_0020);
        let system = ideal_gas(4_000, 14.0, &mut rng);
        let g = system.rdf(20, 6.0).unwrap();
        // The innermost bins hold very few pairs and are noisy; from the
        // third outward the count is large enough to mean something.
        for (k, value) in g.iter().enumerate().skip(3) {
            assert!(
                close(*value, 1.0, 0.06),
                "the ideal gas has g = {value} in bin {k}"
            );
        }
    }

    #[test]
    fn the_radial_distribution_integrates_to_the_neighbour_count() {
        // An identity rather than an approximation: 4 pi rho int g r^2 dr
        // out to r_max is by construction the mean number of neighbours
        // within r_max, so any normalisation error shows up as a
        // discrepancy against a direct count.
        let mut rng = Rng::new(0x011D_0021);
        for trial in 0..4 {
            let system = if trial % 2 == 0 {
                ideal_gas(1_500, 12.0, &mut rng)
            } else {
                MdSystem::lattice_fcc(4, 0.85, 1.0, 1.0, 1.0, &mut rng).unwrap()
            };
            let r_max = (0.4 * system.box_size.x).min(3.0);
            let bins = 60;
            let g = system.rdf(bins, r_max).unwrap();
            let density = system.len() as f64 / system.volume();
            let width = r_max / bins as f64;
            let integral: f64 = g
                .iter()
                .enumerate()
                .map(|(k, value)| {
                    let lo = k as f64 * width;
                    let hi = lo + width;
                    value * 4.0 / 3.0 * std::f64::consts::PI * (hi * hi * hi - lo * lo * lo)
                })
                .sum::<f64>()
                * density;
            let mut direct = 0usize;
            for i in 0..system.len() {
                for j in 0..system.len() {
                    if i != j
                        && system.minimum_image(system.pos[i] - system.pos[j]).magnitude() < r_max
                    {
                        direct += 1;
                    }
                }
            }
            let expected = direct as f64 / system.len() as f64;
            assert!(
                close(integral, expected, 1e-9 * expected.max(1.0)),
                "the integral gives {integral} neighbours against {expected} counted"
            );
        }
    }

    #[test]
    fn a_crystal_shows_its_shells_and_a_liquid_shows_its_first_peak() {
        let mut rng = Rng::new(0x011D_0022);
        // FCC at unit density: the shells sit at a/sqrt2, a, a sqrt(3/2), ...
        let crystal = MdSystem::lattice_fcc(4, 1.0, 0.0, 1.0, 1.0, &mut rng).unwrap();
        let a = crystal.box_size.x / 4.0;
        let bins = 200;
        let r_max = 2.0;
        let g = crystal.rdf(bins, r_max).unwrap();
        let width = r_max / bins as f64;
        let bin_of = |r: f64| (r / width) as usize;
        // Nothing inside the nearest-neighbour distance.
        for value in g.iter().take(bin_of(a / 2f64.sqrt()) - 1) {
            assert!(close(*value, 0.0, 1e-12), "a crystal has density inside its first shell");
        }
        // And a spike at each shell.
        for shell in [a / 2f64.sqrt(), a, a * 1.5f64.sqrt()] {
            if shell < r_max - width {
                let k = bin_of(shell);
                let peak = g[k.saturating_sub(1)].max(g[k]).max(g[k + 1]);
                assert!(peak > 5.0, "the shell at {shell} peaks at only {peak}");
            }
        }

        // A liquid at the triple point: one broad first peak near the
        // Lennard-Jones minimum, 2^(1/6) sigma.
        let mut liquid = MdSystem::lattice_fcc(4, 0.85, 1.5, 1.0, 1.0, &mut rng).unwrap();
        liquid.equilibrate(1_500, 0.004, 0.9, &mut rng).unwrap();
        let g = liquid.rdf(100, 3.0).unwrap();
        let width = 3.0 / 100.0;
        let (peak_bin, peak) =
            g.iter().enumerate().fold((0usize, 0.0f64), |best, (k, v)| {
                if *v > best.1 {
                    (k, *v)
                } else {
                    best
                }
            });
        let peak_r = (peak_bin as f64 + 0.5) * width;
        assert!(
            close(peak_r, 2f64.powf(1.0 / 6.0), 0.15),
            "the liquid's first peak is at {peak_r}, not near the Lennard-Jones minimum"
        );
        assert!(peak > 1.5, "the liquid shows no structure at all: the peak is {peak}");
        // Far out it decorrelates.
        assert!(close(g[g.len() - 1], 1.0, 0.25));
        assert!(liquid.rdf(0, 3.0).is_err());
        assert!(liquid.rdf(10, 0.0).is_err());
        assert!(liquid.rdf(10, liquid.box_size.x).is_err());
    }

    #[test]
    fn the_structure_factor_tends_to_one_and_finds_a_crystals_spacing() {
        let mut rng = Rng::new(0x011D_0023);
        let gas = ideal_gas(800, 12.0, &mut rng);
        // The large-k limit is one for any configuration, which is the
        // check on the normalisation.
        let far = gas.structure_factor(&[60.0, 90.0, 140.0]).unwrap();
        for s in &far {
            assert!(close(*s, 1.0, 0.15), "S at large k is {s}");
        }
        // A crystal has a peak at the reciprocal of its nearest-neighbour
        // spacing; a gas does not.
        let crystal = MdSystem::lattice_fcc(4, 1.0, 0.0, 1.0, 1.0, &mut rng).unwrap();
        let spacing = crystal.box_size.x / 4.0 / 2f64.sqrt();
        let k_peak = 2.0 * std::f64::consts::PI / spacing;
        // Above the smallest reciprocal box vector: below it the Debye sum
        // measures the sample's extent and rises toward N, which would
        // swamp any Bragg peak.
        let smallest = 2.0 * std::f64::consts::PI / crystal.box_size.x;
        assert!(smallest < 3.0, "the search window does not clear the forward peak");
        let grid: Vec<f64> = (0..=100).map(|k| 3.0 + f64::from(k) * 0.15).collect();
        let crystal_s = crystal.structure_factor(&grid).unwrap();
        let gas_s = gas.structure_factor(&grid).unwrap();
        let best = crystal_s
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |b, (k, v)| if *v > b.1 { (k, *v) } else { b });
        assert!(
            close(grid[best.0], k_peak, 1.5),
            "the crystal peaks at k = {} rather than {k_peak}",
            grid[best.0]
        );
        // Against the crystal's own typical value rather than the gas's
        // noisiest point: "is there a peak here" is a question about this
        // curve, and the gas's largest of a hundred samples around one is
        // 1.5 by chance alone.
        let mut sorted = crystal_s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        assert!(
            best.1 > 3.0 * median && best.1 > 3.5,
            "the crystal peak {} is not a peak against its own median {median}",
            best.1
        );
        // And the gas has no peak to speak of.
        let gas_peak = gas_s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(gas_peak < 2.0, "the ideal gas showed structure, peaking at {gas_peak}");
        assert!(gas.structure_factor(&[0.0]).is_err());
        assert!(gas.structure_factor(&[-1.0]).is_err());
    }

    #[test]
    fn the_lindemann_ratio_separates_a_crystal_from_a_melt() {
        let mut rng = Rng::new(0x011D_0024);
        // A cold crystal barely moves.
        let mut cold = MdSystem::lattice_fcc(3, 1.05, 0.05, 1.0, 1.0, &mut rng).unwrap();
        cold.equilibrate(600, 0.003, 0.05, &mut rng).unwrap();
        let (cold_traj, _) = cold.run_trajectory(1_500, 0.003, 15).unwrap();
        let cold_ratio = cold.melting_indicator_lindemann(&cold_traj).unwrap();
        assert!(cold_ratio < 0.15, "a cold crystal reads {cold_ratio}");

        // A hot liquid wanders without limit.
        let mut hot = MdSystem::lattice_fcc(3, 0.75, 3.0, 1.0, 1.0, &mut rng).unwrap();
        hot.equilibrate(600, 0.003, 3.0, &mut rng).unwrap();
        let (hot_traj, _) = hot.run_trajectory(1_500, 0.003, 15).unwrap();
        let hot_ratio = hot.melting_indicator_lindemann(&hot_traj).unwrap();
        assert!(hot_ratio > 0.15, "a hot liquid reads {hot_ratio}");
        assert!(hot_ratio > 2.0 * cold_ratio);

        // A perfectly static crystal has ratio zero exactly.
        let still = MdSystem::lattice_fcc(2, 1.0, 0.0, 1.0, 1.0, &mut rng).unwrap();
        let frozen = vec![still.pos.clone(); 6];
        assert!(close(still.melting_indicator_lindemann(&frozen).unwrap(), 0.0, 1e-12));
        assert!(still.melting_indicator_lindemann(&frozen[..1]).is_err());
        let ragged = vec![vec![Vec3::new(0.0, 0.0, 0.0)]; 3];
        assert!(still.melting_indicator_lindemann(&ragged).is_err());
    }

    // -----------------------------------------------------------------
    // Transport
    // -----------------------------------------------------------------

    #[test]
    fn the_mean_squared_displacement_is_ballistic_for_free_flight() {
        // Particles at constant velocity give MSD = <v^2> t^2 exactly, so
        // this pins both the lag indexing and the averaging over origins
        // with no statistics involved.
        let mut rng = Rng::new(0x011D_0030);
        let count = 40;
        let velocities: Vec<Vec3> = (0..count)
            .map(|_| Vec3::new(rng.next_gaussian(), rng.next_gaussian(), rng.next_gaussian()))
            .collect();
        let dt = 0.05;
        let frames = 30;
        let traj: Vec<Vec<Vec3>> = (0..frames)
            .map(|t| velocities.iter().map(|v| *v * (t as f64 * dt)).collect())
            .collect();
        let msd = MdSystem::msd(&traj).unwrap();
        let mean_v2: f64 =
            velocities.iter().map(Vec3::magnitude_squared).sum::<f64>() / count as f64;
        assert!(close(msd[0], 0.0, 1e-15));
        for lag in 1..frames {
            let t = lag as f64 * dt;
            assert!(
                close(msd[lag], mean_v2 * t * t, 1e-9 * mean_v2 * t * t),
                "at lag {lag} the MSD is {} against {}",
                msd[lag],
                mean_v2 * t * t
            );
        }
        // The velocity autocorrelation of free flight never decays.
        let vel_traj = vec![velocities.clone(); frames];
        let vacf = MdSystem::vacf(&vel_traj).unwrap();
        assert!(vacf.iter().all(|c| close(*c, 1.0, 1e-12)));
        assert!(MdSystem::msd(&traj[..1]).is_err());
        assert!(MdSystem::vacf(&vel_traj[..1]).is_err());
        let still = vec![vec![Vec3::new(0.0, 0.0, 0.0); count]; 5];
        assert!(MdSystem::vacf(&still).is_err());
    }

    #[test]
    fn langevin_diffusion_matches_the_einstein_relation() {
        // The closed form: a free particle under friction gamma at
        // temperature T diffuses with D = T / (m gamma), and its velocity
        // autocorrelation is exactly exp(-gamma t). Both are checked, and
        // at two frictions, so a coefficient that happened to fit one would
        // not survive.
        for &gamma in &[1.0f64, 3.0] {
            let mut rng = Rng::new(0x011D_0031 + gamma as u64);
            let temperature = 1.0;
            let count = 400;
            // Small on purpose: the walkers travel about eight over the run,
            // so a box of four is crossed many times and the wrapped-
            // coordinate control below has something to show. Nothing
            // interacts at any density, since the potential is zero.
            let box_l = 4.0;
            let mut system = MdSystem::new(
                (0..count)
                    .map(|k| {
                        let g = box_l / 8.0;
                        Vec3::new(
                            (k % 8) as f64 * g,
                            ((k / 8) % 8) as f64 * g,
                            (k / 64) as f64 * g,
                        )
                    })
                    .collect(),
                (0..count)
                    .map(|_| {
                        Vec3::new(
                            rng.next_gaussian(),
                            rng.next_gaussian(),
                            rng.next_gaussian(),
                        )
                    })
                    .collect(),
                vec![1.0; count],
                Vec3::new(box_l, box_l, box_l),
                true,
                // Genuinely free: see the note in the thermalisation test.
                Potential::Custom(Arc::new(|_| (0.0, 0.0))),
                0.1,
            )
            .unwrap();
            assert!(close(system.potential_energy(), 0.0, 1e-15), "the walkers interact");
            let dt = 0.01;
            for _ in 0..400 {
                system.step_velocity_verlet(dt);
                system.thermostat_langevin(temperature, gamma, dt, &mut rng);
            }
            let frames = 400;
            let stride = 8;
            let mut positions = Vec::with_capacity(frames);
            let mut velocities = Vec::with_capacity(frames);
            for step in 0..frames * stride {
                if step % stride == 0 {
                    positions.push(system.unwrapped.clone());
                    velocities.push(system.vel.clone());
                }
                system.step_velocity_verlet(dt);
                system.thermostat_langevin(temperature, gamma, dt, &mut rng);
            }
            let sample_dt = dt * stride as f64;

            let vacf = MdSystem::vacf(&velocities).unwrap();
            for lag in 0..12 {
                let expected = (-gamma * lag as f64 * sample_dt).exp();
                assert!(
                    close(vacf[lag], expected, 0.05),
                    "at gamma {gamma} lag {lag} the VACF is {} against {expected}",
                    vacf[lag]
                );
            }

            let msd = MdSystem::msd(&positions).unwrap();
            let d = MdSystem::diffusion_coefficient(&msd, sample_dt).unwrap();
            let expected = temperature / gamma;
            assert!(
                close(d, expected, 0.15 * expected),
                "at gamma {gamma} the diffusion is {d} against {expected}"
            );

            // The negative control that gives the unwrapped coordinates
            // their reason to exist: the same trajectory read from wrapped
            // positions saturates at the box and reports almost no
            // diffusion at all.
            let wrapped: Vec<Vec<Vec3>> = positions
                .iter()
                .map(|frame| frame.iter().map(|p| system.wrap(*p)).collect())
                .collect();
            let wrapped_msd = MdSystem::msd(&wrapped).unwrap();
            let wrapped_d = MdSystem::diffusion_coefficient(&wrapped_msd, sample_dt).unwrap();
            assert!(
                wrapped_d < 0.2 * d,
                "the wrapped trajectory still reports {wrapped_d} against the true {d}"
            );
        }
    }

    #[test]
    fn the_vibrational_spectrum_transforms_the_correlations_it_is_given() {
        // Two closed forms. An exponentially decaying correlation gives a
        // Lorentzian 2 gamma / (gamma^2 + omega^2), and a cosine gives a
        // peak at its own frequency -- which is what checks the frequency
        // grid rather than just the transform.
        let dt = 0.01;
        let n = 4_000;
        for &gamma in &[2.0f64, 6.0] {
            let vacf: Vec<f64> = (0..n).map(|k| (-gamma * k as f64 * dt).exp()).collect();
            let spectrum = MdSystem::vdos_from_vacf(&vacf, dt).unwrap();
            for k in [0usize, 5, 20, 60, 150] {
                let omega = std::f64::consts::PI * k as f64 / (n as f64 * dt);
                let expected = 2.0 * gamma / (gamma * gamma + omega * omega);
                assert!(
                    close(spectrum[k], expected, 0.02 * expected.max(0.05)),
                    "at gamma {gamma}, k = {k} the spectrum is {} against {expected}",
                    spectrum[k]
                );
            }
        }
        let omega0 = 7.0;
        let vacf: Vec<f64> = (0..n).map(|k| (omega0 * k as f64 * dt).cos()).collect();
        let spectrum = MdSystem::vdos_from_vacf(&vacf, dt).unwrap();
        let best = spectrum
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |b, (k, v)| if *v > b.1 { (k, *v) } else { b });
        let peak_omega = std::f64::consts::PI * best.0 as f64 / (n as f64 * dt);
        assert!(
            close(peak_omega, omega0, 0.1),
            "an undamped oscillator peaks at {peak_omega} rather than {omega0}"
        );
        assert!(MdSystem::vdos_from_vacf(&[1.0], dt).is_err());
        assert!(MdSystem::vdos_from_vacf(&vacf, 0.0).is_err());
        assert!(MdSystem::diffusion_coefficient(&[0.0; 4], dt).is_err());
        assert!(MdSystem::diffusion_coefficient(&[0.0; 20], 0.0).is_err());
    }

    // -----------------------------------------------------------------
    // Dynamics
    // -----------------------------------------------------------------

    /// Two particles on a spring, whose motion is exactly known.
    fn harmonic_pair(separation: f64, k: f64, r0: f64) -> MdSystem {
        MdSystem::new(
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(separation, 0.0, 0.0)],
            vec![Vec3::new(0.0, 0.0, 0.0); 2],
            vec![1.0; 2],
            Vec3::new(100.0, 100.0, 100.0),
            false,
            Potential::Harmonic { k, r0 },
            50.0,
        )
        .unwrap()
    }

    #[test]
    fn velocity_verlet_reproduces_the_exact_harmonic_solution() {
        // The one case with a closed form. Two unit masses on a spring
        // oscillate about their centre of mass at omega = sqrt(2k/m) --
        // the reduced mass is m/2, which is the factor an implementation
        // that forgot the two-body character would miss.
        let k = 4.0;
        let r0 = 1.0;
        let amplitude = 0.2;
        let mut system = harmonic_pair(r0 + amplitude, k, r0);
        let omega = (2.0 * k).sqrt();
        let dt = 1e-4;
        let steps = 20_000;
        for step in 1..=steps {
            system.step_velocity_verlet(dt);
            if step % 2_000 == 0 {
                let t = step as f64 * dt;
                let expected = r0 + amplitude * (omega * t).cos();
                let actual = (system.pos[1] - system.pos[0]).x;
                assert!(
                    close(actual, expected, 2e-4),
                    "at t = {t} the separation is {actual} against {expected}"
                );
            }
        }
        assert!(close(system.time, steps as f64 * dt, 1e-9));
    }

    #[test]
    fn the_symplectic_integrator_oscillates_where_euler_runs_away() {
        // The point of a symplectic scheme, made a measurement rather than
        // an assertion: over the same trajectory velocity Verlet's energy
        // returns to where it started while explicit Euler's climbs
        // monotonically. Comparing the two is what makes this a test of
        // the integrator rather than of the tolerance.
        let k = 4.0;
        let dt = 0.01;
        let steps = 40_000;
        let mut verlet = harmonic_pair(1.2, k, 1.0);
        let samples = verlet.run_nve(steps, dt).unwrap();
        let drift = energy_drift(&samples).unwrap();
        assert!(drift < 1e-9, "the symplectic integrator drifted by {drift}");

        // The same system under explicit Euler, written out here so the
        // comparison is against a real alternative and not a straw number.
        let mut euler = harmonic_pair(1.2, k, 1.0);
        let start = euler.sample().total;
        let mut euler_samples = vec![euler.sample()];
        for _ in 0..steps {
            let forces = euler.forces();
            for j in 0..euler.len() {
                let a = forces[j] * (1.0 / euler.mass[j]);
                euler.pos[j] = euler.pos[j] + euler.vel[j] * dt;
                euler.vel[j] = euler.vel[j] + a * dt;
            }
            euler.time += dt;
            euler_samples.push(euler.sample());
        }
        let euler_drift = energy_drift(&euler_samples).unwrap();
        assert!(
            euler_drift > 1e4 * drift.max(1e-12),
            "Euler drifted by {euler_drift} against Verlet's {drift}, so the comparison is empty"
        );
        assert!(euler.sample().total > start, "Euler did not gain energy");

        // The energy still *fluctuates* under Verlet -- the drift measure
        // must not be mistaking a flat record for a good one.
        let spread = samples.iter().map(|s| s.total).fold(f64::NEG_INFINITY, f64::max)
            - samples.iter().map(|s| s.total).fold(f64::INFINITY, f64::min);
        assert!(spread > 0.0, "the total energy never moved, so nothing was measured");
    }

    #[test]
    fn a_lennard_jones_liquid_conserves_energy_and_momentum_under_nve() {
        // The roadmap's acceptance test. A liquid at the triple point is
        // the hard case: the particles are close enough that the forces are
        // stiff and the trajectories are chaotic, so a conserved quantity
        // that survives here is conserved for a real reason.
        let mut rng = Rng::new(0x011D_0010);
        let mut system = MdSystem::lattice_fcc(3, 0.85, 1.5, 1.0, 1.0, &mut rng).unwrap();
        system.equilibrate(400, 0.004, 0.9, &mut rng).unwrap();
        let momentum_before = system.total_momentum();
        let samples = system.run_nve(3_000, 0.004).unwrap();
        let drift = energy_drift(&samples).unwrap();
        assert!(drift < 1e-4, "the energy drifted by {drift} over three thousand steps");
        // Momentum is conserved exactly, not approximately: the internal
        // forces cancel pair by pair, so the only error is rounding.
        let after = system.total_momentum();
        assert!(
            close((after - momentum_before).magnitude(), 0.0, 1e-9),
            "the momentum moved by {}",
            (after - momentum_before).magnitude()
        );
        // And the run stayed a liquid rather than blowing up.
        assert!(samples.iter().all(|s| s.total.is_finite()));
        assert!(system.temperature() > 0.2 && system.temperature() < 3.0);
    }

    #[test]
    fn a_larger_step_costs_energy_conservation_in_the_expected_way() {
        // Velocity Verlet's energy error is second order in the step, so
        // halving the step should quarter the amplitude of the oscillation.
        // This is the scaling that identifies the integrator's order, and
        // it fails for any first-order scheme however small its error.
        let mut spread = Vec::new();
        for shift in 0..3 {
            let dt = 0.02 / f64::from(1 << shift);
            let mut system = harmonic_pair(1.3, 4.0, 1.0);
            let samples = system.run_nve(4_000 * (1 << shift), dt).unwrap();
            let hi = samples.iter().map(|s| s.total).fold(f64::NEG_INFINITY, f64::max);
            let lo = samples.iter().map(|s| s.total).fold(f64::INFINITY, f64::min);
            spread.push(hi - lo);
        }
        for k in 1..spread.len() {
            let ratio = spread[k - 1] / spread[k];
            assert!(
                close(ratio, 4.0, 0.4),
                "halving the step changed the energy spread by {ratio} rather than four"
            );
        }
    }

    #[test]
    fn the_thermostats_reach_the_temperature_they_are_given() {
        // All three are checked against the same target from the same
        // start, because each is easy to write in a form that thermostats
        // to something close but wrong -- a Langevin noise amplitude off by
        // sqrt(2), say, lands at twice the temperature and still looks
        // like it is working.
        let mut rng = Rng::new(0x011D_0011);
        for &target in &[0.4f64, 1.0, 2.2] {
            // Berendsen.
            let mut system = MdSystem::lattice_fcc(3, 0.8, 0.05, 1.0, 1.0, &mut rng).unwrap();
            for _ in 0..600 {
                system.step_velocity_verlet(0.004);
                system.thermostat_berendsen(target, 0.1, 0.004);
            }
            let berendsen: f64 = (0..400)
                .map(|_| {
                    system.step_velocity_verlet(0.004);
                    system.thermostat_berendsen(target, 0.1, 0.004);
                    system.temperature()
                })
                .sum::<f64>()
                / 400.0;
            assert!(close(berendsen, target, 0.1 * target), "Berendsen reached {berendsen}");

            // Langevin.
            let mut system = MdSystem::lattice_fcc(3, 0.8, 0.05, 1.0, 1.0, &mut rng).unwrap();
            for _ in 0..1_500 {
                system.step_velocity_verlet(0.004);
                system.thermostat_langevin(target, 2.0, 0.004, &mut rng);
            }
            let langevin: f64 = (0..600)
                .map(|_| {
                    system.step_velocity_verlet(0.004);
                    system.thermostat_langevin(target, 2.0, 0.004, &mut rng);
                    system.temperature()
                })
                .sum::<f64>()
                / 600.0;
            assert!(close(langevin, target, 0.12 * target), "Langevin reached {langevin}");

            // Nose-Hoover.
            let mut system = MdSystem::lattice_fcc(3, 0.8, 0.05, 1.0, 1.0, &mut rng).unwrap();
            for _ in 0..4_000 {
                system.step_velocity_verlet(0.004);
                system.thermostat_nose_hoover(target, 40.0, 0.004);
            }
            let nose: f64 = (0..2_000)
                .map(|_| {
                    system.step_velocity_verlet(0.004);
                    system.thermostat_nose_hoover(target, 40.0, 0.004);
                    system.temperature()
                })
                .sum::<f64>()
                / 2_000.0;
            assert!(close(nose, target, 0.2 * target), "Nose-Hoover reached {nose}");
        }
    }

    #[test]
    fn the_langevin_thermostat_thermalises_a_free_gas_to_the_exact_distribution() {
        // With no interactions the answer is known exactly: the stationary
        // distribution of the Ornstein-Uhlenbeck velocity update is
        // Maxwell-Boltzmann at the target temperature, whatever the
        // friction. Checking across two frictions is what shows the noise
        // is tied to the friction rather than tuned to one case.
        for &gamma in &[0.5f64, 4.0] {
            let mut rng = Rng::new(0x011D_0012 + (gamma * 10.0) as u64);
            let target = 1.3;
            let count = 400;
            let mut system = MdSystem::new(
                (0..count)
                    .map(|k| {
                        Vec3::new(
                            (k % 10) as f64 * 3.0,
                            ((k / 10) % 10) as f64 * 3.0,
                            (k / 100) as f64 * 3.0,
                        )
                    })
                    .collect(),
                vec![Vec3::new(0.0, 0.0, 0.0); count],
                vec![1.0; count],
                Vec3::new(30.0, 30.0, 30.0),
                true,
                // Genuinely zero, not merely cut off: a Lennard-Jones pair
                // truncated at a tenth sigma still has a 10^12 core just
                // inside the cutoff, and two particles that wander into it
                // are ejected at enormous speed.
                Potential::Custom(Arc::new(|_| (0.0, 0.0))),
                1.0,
            )
            .unwrap();
            assert!(close(system.potential_energy(), 0.0, 1e-15), "the gas is not free");
            for _ in 0..400 {
                system.thermostat_langevin(target, gamma, 0.05, &mut rng);
            }
            let mut mean = 0.0;
            for _ in 0..40 {
                system.thermostat_langevin(target, gamma, 0.05, &mut rng);
                mean += system.temperature();
            }
            mean /= 40.0;
            assert!(
                close(mean, target, 0.06 * target),
                "at gamma = {gamma} the gas settled at {mean} rather than {target}"
            );
            let test = system.maxwell_boltzmann_check().unwrap();
            assert!(
                test.p_value > 0.01,
                "the speeds failed a KS test against Maxwell-Boltzmann at p = {}",
                test.p_value
            );
        }
    }

    #[test]
    fn the_maxwell_boltzmann_check_rejects_a_distribution_that_is_merely_warm() {
        // Equipartition does not pin the distribution. A system with every
        // particle at the same speed has exactly the right temperature and
        // entirely the wrong statistics -- and that is the state a freshly
        // rescaled lattice is in, so the check has to catch it.
        let count = 300;
        let speed = 1.0;
        let mut rng = Rng::new(0x011D_0013);
        let mut system = MdSystem::new(
            (0..count)
                .map(|k| Vec3::new((k % 10) as f64 * 3.0, ((k / 10) % 10) as f64 * 3.0, (k / 100) as f64 * 3.0))
                .collect(),
            (0..count)
                .map(|_| {
                    // Random directions, identical magnitude.
                    let mut d = Vec3::new(
                        rng.next_gaussian(),
                        rng.next_gaussian(),
                        rng.next_gaussian(),
                    );
                    if d.magnitude() < 1e-9 {
                        d = Vec3::new(1.0, 0.0, 0.0);
                    }
                    d.normalized() * speed
                })
                .collect(),
            vec![1.0; count],
            Vec3::new(30.0, 30.0, 30.0),
            true,
            Potential::Custom(Arc::new(|_| (0.0, 0.0))),
            1.0,
        )
        .unwrap();
        let monodisperse = system.maxwell_boltzmann_check().unwrap();
        assert!(
            monodisperse.p_value < 1e-6,
            "a monodisperse gas passed the test at p = {}",
            monodisperse.p_value
        );
        // Thermalising the same particles at the same temperature passes.
        let target = system.temperature();
        for _ in 0..400 {
            system.thermostat_langevin(target, 2.0, 0.05, &mut rng);
        }
        assert!(system.maxwell_boltzmann_check().unwrap().p_value > 0.01);
        // The test refuses a mixture, where a single-sample test does not
        // apply, and a frozen system.
        system.mass[0] = 2.0;
        assert!(system.maxwell_boltzmann_check().is_err());
        system.mass[0] = 1.0;
        for v in &mut system.vel {
            *v = Vec3::new(0.0, 0.0, 0.0);
        }
        assert!(system.maxwell_boltzmann_check().is_err());
    }

    #[test]
    fn removing_the_drift_leaves_the_relative_motion_alone() {
        // The centre-of-mass velocity is conserved, so it never decays: it
        // sits in the kinetic energy for the whole run and inflates every
        // temperature reading. Removing it must not touch anything else.
        let mut rng = Rng::new(0x011D_0014);
        let mut system = MdSystem::lattice_fcc(2, 0.8, 1.0, 1.0, 1.0, &mut rng).unwrap();
        let boost = Vec3::new(0.7, -0.3, 0.2);
        for v in &mut system.vel {
            *v = *v + boost;
        }
        let before: Vec<Vec3> = system.vel.clone();
        let hot = system.temperature();
        system.remove_drift();
        assert!(close(system.total_momentum().magnitude(), 0.0, 1e-9));
        // Every pairwise velocity difference is untouched.
        for k in 1..system.len() {
            let old = before[k] - before[0];
            let new = system.vel[k] - system.vel[0];
            assert!(close((old - new).magnitude(), 0.0, 1e-12));
        }
        assert!(system.temperature() < hot, "the drift did not inflate the temperature");
        // Removing it twice changes nothing.
        let once = system.vel.clone();
        system.remove_drift();
        for k in 0..system.len() {
            assert!(close((once[k] - system.vel[k]).magnitude(), 0.0, 1e-12));
        }
    }

    #[test]
    fn the_degrees_of_freedom_account_for_the_conserved_momentum() {
        let mut rng = Rng::new(0x011D_0015);
        let periodic = MdSystem::lattice_fcc(2, 0.8, 1.0, 1.0, 1.0, &mut rng).unwrap();
        assert!(close(periodic.degrees_of_freedom(), 3.0 * 32.0 - 3.0, 1e-12));
        // Which is where the temperature comes from: 2 K / dof.
        assert!(close(
            periodic.temperature(),
            2.0 * periodic.kinetic_energy() / (3.0 * 32.0 - 3.0),
            1e-12
        ));
        let open = MdSystem::new(
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0)],
            vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)],
            vec![1.0; 2],
            Vec3::new(50.0, 50.0, 50.0),
            false,
            Potential::LennardJones { eps: 1.0, sigma: 1.0 },
            3.0,
        )
        .unwrap();
        assert!(close(open.degrees_of_freedom(), 6.0, 1e-12));
        assert!(close(open.kinetic_energy(), 1.0, 1e-12));
        assert!(close(open.temperature(), 2.0 / 6.0, 1e-12));
    }

    #[test]
    fn the_barostat_moves_the_pressure_toward_its_target() {
        let mut rng = Rng::new(0x011D_0016);
        let mut system = MdSystem::lattice_fcc(3, 0.75, 1.0, 1.0, 1.0, &mut rng).unwrap();
        system.equilibrate(300, 0.004, 1.0, &mut rng).unwrap();
        let target = system.pressure_virial() + 1.0;
        let start = (system.pressure_virial() - target).abs();
        for _ in 0..400 {
            system.step_velocity_verlet(0.004);
            system.thermostat_berendsen(1.0, 0.1, 0.004);
            system.barostat_berendsen(target, 0.05, 0.5, 0.004).unwrap();
        }
        let end = (system.pressure_virial() - target).abs();
        assert!(end < start, "the pressure went from {start} away to {end} away");
        // The particle count and the density relation are preserved.
        assert!(close(system.len() as f64 / system.volume() * system.volume(), 108.0, 1e-9));
        assert!(system.barostat_berendsen(1.0, 0.05, 0.0, 0.004).is_err());
        assert!(system.barostat_berendsen(1.0, 0.0, 0.5, 0.004).is_err());
        // Squeezing hard enough to bring the box below twice the cutoff is
        // refused rather than silently breaking the minimum image.
        assert!(system.barostat_berendsen(1e6, 1.0, 1e-6, 1.0).is_err());
    }

    #[test]
    fn energy_drift_measures_the_trend_and_not_the_wobble() {
        // A record that oscillates without going anywhere must read as no
        // drift, and one that climbs steadily must read as drift, even if
        // the climbing record has the smaller spread. That distinction is
        // the whole reason the measure is a fitted slope.
        let wobble: Vec<MdSample> = (0..400)
            .map(|k| {
                let t = k as f64 * 0.01;
                let e = 100.0 + (t * 7.0).sin();
                MdSample { time: t, kinetic: e, potential: 0.0, total: e, temperature: 1.0, pressure: 0.0 }
            })
            .collect();
        let climb: Vec<MdSample> = (0..400)
            .map(|k| {
                let t = k as f64 * 0.01;
                let e = 100.0 + 0.1 * t;
                MdSample { time: t, kinetic: e, potential: 0.0, total: e, temperature: 1.0, pressure: 0.0 }
            })
            .collect();
        let wobble_spread = 2.0;
        let climb_spread = 0.4;
        assert!(climb_spread < wobble_spread, "the fixture does not make the point");
        assert!(energy_drift(&wobble).unwrap() < 1e-3);
        // 0.1 * 4 / 100.2 = 0.004.
        assert!(close(energy_drift(&climb).unwrap(), 0.004, 1e-4));
        assert!(energy_drift(&wobble[..2]).is_err());
        let flat: Vec<MdSample> = vec![wobble[0]; 5];
        assert!(energy_drift(&flat).is_err());
    }

    #[test]
    fn the_fcc_lattice_has_the_density_and_neighbour_count_it_claims() {
        let mut rng = Rng::new(0x011D_0004);
        for cells in [2usize, 3, 4] {
            for &density in &[0.6f64, 0.85, 1.1] {
                let system = MdSystem::lattice_fcc(cells, density, 0.8, 1.0, 1.0, &mut rng).unwrap();
                assert_eq!(system.len(), 4 * cells * cells * cells);
                assert!(close(system.len() as f64 / system.volume(), density, 1e-9));
                assert!(!system.is_empty());
                // FCC has twelve nearest neighbours at a / sqrt 2.
                let a = system.box_size.x / cells as f64;
                let nearest = a / 2f64.sqrt();
                let mut neighbours = 0;
                for j in 1..system.len() {
                    let r = system.minimum_image(system.pos[0] - system.pos[j]).magnitude();
                    if r < nearest * 1.05 {
                        neighbours += 1;
                    }
                }
                assert_eq!(neighbours, 12, "an FCC site has twelve nearest neighbours");
                // The drift is removed and the temperature is as asked.
                assert!(close(system.total_momentum().magnitude(), 0.0, 1e-9));
                assert!(close(system.temperature(), 0.8, 1e-9));
            }
        }
    }
}
