use crate::math::Vec3;

pub fn hill_radius(distance: f64, body_mass: f64, primary_mass: f64) -> f64 {
    if primary_mass <= 0.0 {
        return 0.0;
    }
    distance * (body_mass / (3.0 * primary_mass)).cbrt()
}

pub fn lagrange_points(
    primary_pos: Vec3,
    primary_mass: f64,
    body_pos: Vec3,
    body_vel: Vec3,
    body_mass: f64,
) -> [Vec3; 5] {
    let delta = body_pos - primary_pos;
    let d = delta.magnitude();
    if d < 1e-20 {
        return [Vec3::ZERO; 5];
    }

    let u = delta * (1.0 / d);
    let r_hill = hill_radius(d, body_mass, primary_mass);

    // L1: between primary and body
    let l1 = body_pos - u * r_hill;
    // L2: beyond body
    let l2 = body_pos + u * r_hill;
    // L3: opposite side of primary
    let l3 = primary_pos - u * d;

    // Orbital plane normal from angular momentum
    let rel_vel = body_vel;
    let h = delta.cross(&rel_vel);
    let h_mag = h.magnitude();

    let h_hat = if h_mag > 1e-20 {
        h * (1.0 / h_mag)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };

    // Perpendicular in orbital plane
    let p = h_hat.cross(&u);

    const COS60: f64 = 0.5;
    const SIN60: f64 = 0.866_025_403_784;

    // L4: 60° ahead
    let l4 = primary_pos + u * (d * COS60) + p * (d * SIN60);
    // L5: 60° behind
    let l5 = primary_pos + u * (d * COS60) - p * (d * SIN60);

    [l1, l2, l3, l4, l5]
}

pub fn escape_velocity(mu: f64, distance: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    (2.0 * mu / distance).sqrt()
}

pub fn circular_velocity(mu: f64, distance: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    (mu / distance).sqrt()
}

pub fn escape_ratio(speed: f64, escape_vel: f64) -> f64 {
    if escape_vel <= 0.0 {
        return 0.0;
    }
    speed / escape_vel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::constants::G;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_hill_radius() {
        // Earth-Sun system: d=1AU, m_earth/m_sun ~ 3e-6
        let r = hill_radius(1.496e11, 5.972e24, 1.989e30);
        // Should be ~1.5 million km
        assert!(r > 1e9 && r < 2e9, "Hill radius = {r}");
    }

    #[test]
    fn test_lagrange_l1_between() {
        let primary = Vec3::ZERO;
        let body = Vec3::new(10.0, 0.0, 0.0);
        let vel = Vec3::new(0.0, 1.0, 0.0);
        let points = lagrange_points(primary, 1000.0, body, vel, 1.0);
        // L1 should be between primary and body
        assert!(points[0].x > 0.0 && points[0].x < 10.0);
    }

    #[test]
    fn test_lagrange_l2_beyond() {
        let primary = Vec3::ZERO;
        let body = Vec3::new(10.0, 0.0, 0.0);
        let vel = Vec3::new(0.0, 1.0, 0.0);
        let points = lagrange_points(primary, 1000.0, body, vel, 1.0);
        // L2 should be beyond body
        assert!(points[1].x > 10.0);
    }

    #[test]
    fn test_lagrange_l4_l5_symmetric() {
        let primary = Vec3::ZERO;
        let body = Vec3::new(10.0, 0.0, 0.0);
        let vel = Vec3::new(0.0, 0.0, 1.0);
        let points = lagrange_points(primary, 1000.0, body, vel, 1.0);
        // L4 and L5 should be at same distance from primary
        let d4 = points[3].magnitude();
        let d5 = points[4].magnitude();
        assert!(approx(d4, d5, 1e-9), "L4 dist={d4}, L5 dist={d5}");
        // Both at distance d from primary
        assert!(approx(d4, 10.0, 1e-6));
    }

    #[test]
    fn test_escape_vs_circular() {
        let mu = G * 1.989e30;
        let r = 1.496e11;
        let v_esc = escape_velocity(mu, r);
        let v_circ = circular_velocity(mu, r);
        assert!(approx(v_esc / v_circ, 2.0_f64.sqrt(), 1e-6));
    }
}
