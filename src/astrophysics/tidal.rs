use crate::math::Vec3;
use crate::math::constants::G;

pub fn tidal_acceleration_magnitude(primary_mass: f64, distance: f64) -> f64 {
    2.0 * G * primary_mass / (distance * distance * distance)
}

pub fn tidal_acceleration_gr_corrected(
    primary_mass: f64,
    distance: f64,
    schwarzschild_radius: f64,
) -> f64 {
    let base = tidal_acceleration_magnitude(primary_mass, distance);
    if distance > schwarzschild_radius * 1.02 {
        base / (1.0 - schwarzschild_radius / distance)
    } else {
        base
    }
}

pub fn roche_limit_rigid(primary_radius: f64, primary_density: f64, satellite_density: f64) -> f64 {
    if satellite_density <= 0.0 {
        return f64::INFINITY;
    }
    primary_radius * (2.0 * primary_density / satellite_density).cbrt()
}

pub fn roche_limit_fluid(primary_radius: f64, primary_density: f64, satellite_density: f64) -> f64 {
    if satellite_density <= 0.0 {
        return f64::INFINITY;
    }
    2.44 * primary_radius * (primary_density / satellite_density).cbrt()
}

pub fn tidal_force_ratio(
    primary_mass: f64,
    body_mass: f64,
    body_radius: f64,
    distance: f64,
) -> f64 {
    let tidal = tidal_acceleration_magnitude(primary_mass, distance) * body_radius;
    let self_gravity = G * body_mass / (body_radius * body_radius);
    if self_gravity <= 0.0 {
        return f64::INFINITY;
    }
    tidal / self_gravity
}

pub fn tidal_tensor_eigenvalues(primary_mass: f64, distance: f64) -> (f64, f64) {
    let base = G * primary_mass / (distance * distance * distance);
    (2.0 * base, -base)
}

pub fn roche_potential(
    x: f64,
    z: f64,
    m1: f64,
    pos1: Vec3,
    m2: f64,
    pos2: Vec3,
) -> f64 {
    let dx = pos2.x - pos1.x;
    let dz = pos2.z - pos1.z;
    let d = (dx * dx + dz * dz).sqrt();
    if d < 1e-20 {
        return 0.0;
    }

    let total_mass = m1 + m2;
    let omega_sq = G * total_mass / (d * d * d);

    let bcx = (pos1.x * m1 + pos2.x * m2) / total_mass;
    let bcz = (pos1.z * m1 + pos2.z * m2) / total_mass;

    let r1 = ((x - pos1.x).powi(2) + (z - pos1.z).powi(2)).sqrt() + 1e-6;
    let r2 = ((x - pos2.x).powi(2) + (z - pos2.z).powi(2)).sqrt() + 1e-6;
    let r_com_sq = (x - bcx).powi(2) + (z - bcz).powi(2);

    -G * m1 / r1 - G * m2 / r2 - 0.5 * omega_sq * r_com_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_tidal_acceleration() {
        let a = tidal_acceleration_magnitude(1.989e30, 1.496e11);
        assert!(a > 0.0);
    }

    #[test]
    fn test_tidal_tensor_ratio() {
        let (radial, tangential) = tidal_tensor_eigenvalues(1.989e30, 1.496e11);
        assert!(approx(radial, -2.0 * tangential, radial.abs() * 1e-10));
    }

    #[test]
    fn test_roche_limit_fluid_gt_rigid() {
        let rigid = roche_limit_rigid(1.0, 5000.0, 3000.0);
        let fluid = roche_limit_fluid(1.0, 5000.0, 3000.0);
        assert!(fluid > rigid, "Fluid Roche limit should exceed rigid");
    }

    #[test]
    fn test_gr_correction_increases_tidal() {
        let rs = 2.0 * G * 1.989e30 / (3e8 * 3e8);
        let r = rs * 10.0;
        let newtonian = tidal_acceleration_magnitude(1.989e30, r);
        let gr = tidal_acceleration_gr_corrected(1.989e30, r, rs);
        assert!(gr > newtonian, "GR correction should increase tidal force");
    }

    #[test]
    fn test_roche_potential_symmetry() {
        let p1 = roche_potential(
            0.5, 0.1,
            1.0, Vec3::new(0.0, 0.0, 0.0),
            1.0, Vec3::new(1.0, 0.0, 0.0),
        );
        let p2 = roche_potential(
            0.5, -0.1,
            1.0, Vec3::new(0.0, 0.0, 0.0),
            1.0, Vec3::new(1.0, 0.0, 0.0),
        );
        assert!(approx(p1, p2, 1e-6), "Potential should be symmetric about line of centers");
    }
}
