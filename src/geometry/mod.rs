use crate::math::constants::PI;

// --- 2D Areas ---

/// Area of a circle: A = πr²
#[must_use]
pub fn area_circle(radius: f64) -> f64 {
    PI * radius * radius
}

/// Area of an ellipse: A = πab
#[must_use]
pub fn area_ellipse(semi_major: f64, semi_minor: f64) -> f64 {
    PI * semi_major * semi_minor
}

/// Area of a triangle: A = bh/2
#[must_use]
pub fn area_triangle(base: f64, height: f64) -> f64 {
    base * height / 2.0
}

/// Area of a triangle via Heron's formula: A = √(s(s-a)(s-b)(s-c)) where s = (a+b+c)/2
#[must_use]
pub fn area_triangle_heron(a: f64, b: f64, c: f64) -> f64 {
    let s = (a + b + c) / 2.0;
    (s * (s - a) * (s - b) * (s - c)).sqrt()
}

/// Area of a regular polygon: A = n·s²/(4·tan(π/n))
#[must_use]
pub fn area_regular_polygon(n_sides: u32, side_length: f64) -> f64 {
    assert!(n_sides >= 3, "polygon must have at least 3 sides");
    let n = f64::from(n_sides);
    (n * side_length * side_length) / (4.0 * (PI / n).tan())
}

/// Area of a circular sector: A = r²θ/2
#[must_use]
pub fn area_sector(radius: f64, angle: f64) -> f64 {
    radius * radius * angle / 2.0
}

/// Area of an annulus: A = π(R² - r²)
#[must_use]
pub fn area_annulus(outer_r: f64, inner_r: f64) -> f64 {
    PI * (outer_r * outer_r - inner_r * inner_r)
}

// --- 3D Volumes ---

/// Volume of a sphere: V = 4πr³/3
#[must_use]
pub fn volume_sphere(radius: f64) -> f64 {
    4.0 * PI * radius * radius * radius / 3.0
}

/// Volume of a cylinder: V = πr²h
#[must_use]
pub fn volume_cylinder(radius: f64, height: f64) -> f64 {
    PI * radius * radius * height
}

/// Volume of a cone: V = πr²h/3
#[must_use]
pub fn volume_cone(radius: f64, height: f64) -> f64 {
    PI * radius * radius * height / 3.0
}

/// Volume of an ellipsoid: V = 4πabc/3
#[must_use]
pub fn volume_ellipsoid(a: f64, b: f64, c: f64) -> f64 {
    4.0 * PI * a * b * c / 3.0
}

/// Volume of a torus: V = 2π²Rr²
#[must_use]
pub fn volume_torus(major_r: f64, minor_r: f64) -> f64 {
    2.0 * PI * PI * major_r * minor_r * minor_r
}

/// Volume of a frustum: V = πh(r₁² + r₁r₂ + r₂²)/3
#[must_use]
pub fn volume_frustum(r1: f64, r2: f64, height: f64) -> f64 {
    PI * height * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0
}

/// Volume of a capsule (cylinder + sphere): V = πr²h + 4πr³/3
#[must_use]
pub fn volume_capsule(radius: f64, cylinder_height: f64) -> f64 {
    PI * radius * radius * cylinder_height + 4.0 * PI * radius * radius * radius / 3.0
}

// --- Surface Areas ---

/// Surface area of a sphere: A = 4πr²
#[must_use]
pub fn surface_sphere(radius: f64) -> f64 {
    4.0 * PI * radius * radius
}

/// Total surface area of a cylinder (lateral + both caps): A = 2πr(r + h)
#[must_use]
pub fn surface_cylinder_total(radius: f64, height: f64) -> f64 {
    2.0 * PI * radius * (radius + height)
}

/// Lateral surface area of a cylinder: A = 2πrh
#[must_use]
pub fn surface_cylinder_lateral(radius: f64, height: f64) -> f64 {
    2.0 * PI * radius * height
}

/// Lateral surface area of a cone: A = πrl
#[must_use]
pub fn surface_cone_lateral(radius: f64, slant_height: f64) -> f64 {
    PI * radius * slant_height
}

/// Surface area of a torus: A = 4π²Rr
#[must_use]
pub fn surface_torus(major_r: f64, minor_r: f64) -> f64 {
    4.0 * PI * PI * major_r * minor_r
}

// --- Solid Angles & Spherical Geometry ---

/// Solid angle subtended by a cone: Ω = 2π(1 - cos(θ))
#[must_use]
pub fn solid_angle_cone(half_angle: f64) -> f64 {
    2.0 * PI * (1.0 - half_angle.cos())
}

/// Solid angle of a full sphere: Ω = 4π steradians
#[must_use]
pub fn solid_angle_full_sphere() -> f64 {
    4.0 * PI
}

/// Great-circle distance on a sphere: d = r·arccos(sin(φ₁)sin(φ₂) + cos(φ₁)cos(φ₂)cos(Δλ))
#[must_use]
pub fn great_circle_distance(r: f64, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let delta_lon = lon2 - lon1;
    let cos_angle = lat1.sin() * lat2.sin() + lat1.cos() * lat2.cos() * delta_lon.cos();
    r * cos_angle.clamp(-1.0, 1.0).acos()
}

/// Spherical excess of a spherical triangle: E = A + B + C - π
#[must_use]
pub fn spherical_excess(a: f64, b: f64, c: f64) -> f64 {
    a + b + c - PI
}

// --- Moments of Inertia ---

/// Moment of inertia of a solid sphere: I = 2mr²/5
#[must_use]
pub fn moi_solid_sphere(mass: f64, radius: f64) -> f64 {
    2.0 * mass * radius * radius / 5.0
}

/// Moment of inertia of a hollow sphere (thin shell): I = 2mr²/3
#[must_use]
pub fn moi_hollow_sphere(mass: f64, radius: f64) -> f64 {
    2.0 * mass * radius * radius / 3.0
}

/// Moment of inertia of a solid cylinder about its axis: I = mr²/2
#[must_use]
pub fn moi_solid_cylinder(mass: f64, radius: f64) -> f64 {
    mass * radius * radius / 2.0
}

/// Moment of inertia of a thin rod about its center: I = mL²/12
#[must_use]
pub fn moi_thin_rod_center(mass: f64, length: f64) -> f64 {
    mass * length * length / 12.0
}

/// Moment of inertia of a thin rod about one end: I = mL²/3
#[must_use]
pub fn moi_thin_rod_end(mass: f64, length: f64) -> f64 {
    mass * length * length / 3.0
}

/// Moment of inertia of a rectangular plate about its center: I = m(w² + h²)/12
#[must_use]
pub fn moi_rectangular_plate(mass: f64, width: f64, height: f64) -> f64 {
    mass * (width * width + height * height) / 12.0
}

/// Parallel axis theorem: I = I_cm + md²
#[must_use]
pub fn parallel_axis(i_cm: f64, mass: f64, distance: f64) -> f64 {
    i_cm + mass * distance * distance
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_rel(a: f64, b: f64) -> bool {
        if b.abs() < EPSILON {
            return a.abs() < EPSILON;
        }
        ((a - b) / b).abs() < 1e-6
    }

    // 2D Areas

    #[test]
    fn test_area_circle() {
        assert!(approx(area_circle(1.0), PI));
        assert!(approx(area_circle(3.0), PI * 9.0));
    }

    #[test]
    fn test_area_ellipse() {
        assert!(approx(area_ellipse(1.0, 1.0), PI));
        assert!(approx(area_ellipse(3.0, 2.0), PI * 6.0));
    }

    #[test]
    fn test_area_triangle() {
        assert!(approx(area_triangle(6.0, 4.0), 12.0));
    }

    #[test]
    fn test_area_triangle_heron() {
        // 3-4-5 right triangle: area = 6
        assert!(approx(area_triangle_heron(3.0, 4.0, 5.0), 6.0));
        // equilateral triangle side 2: area = sqrt(3)
        assert!(approx_rel(area_triangle_heron(2.0, 2.0, 2.0), 3.0_f64.sqrt()));
    }

    #[test]
    fn test_area_regular_polygon() {
        // Square with side 2: area = 4
        assert!(approx_rel(area_regular_polygon(4, 2.0), 4.0));
        // Equilateral triangle side 2: area = sqrt(3)
        assert!(approx_rel(area_regular_polygon(3, 2.0), 3.0_f64.sqrt()));
    }

    #[test]
    fn test_area_sector() {
        // Full circle: angle = 2π, should equal πr²
        assert!(approx(area_sector(3.0, 2.0 * PI), PI * 9.0));
        // Half circle
        assert!(approx(area_sector(2.0, PI), PI * 2.0));
    }

    #[test]
    fn test_area_annulus() {
        assert!(approx(area_annulus(3.0, 2.0), PI * 5.0));
        assert!(approx(area_annulus(1.0, 0.0), PI));
    }

    // 3D Volumes

    #[test]
    fn test_volume_sphere() {
        assert!(approx_rel(volume_sphere(1.0), 4.0 * PI / 3.0));
    }

    #[test]
    fn test_volume_cylinder() {
        assert!(approx(volume_cylinder(2.0, 5.0), PI * 4.0 * 5.0));
    }

    #[test]
    fn test_volume_cone() {
        assert!(approx(volume_cone(3.0, 6.0), PI * 9.0 * 6.0 / 3.0));
    }

    #[test]
    fn test_volume_ellipsoid() {
        // Sphere case: a=b=c=r
        assert!(approx(volume_ellipsoid(2.0, 2.0, 2.0), volume_sphere(2.0)));
    }

    #[test]
    fn test_volume_torus() {
        assert!(approx(volume_torus(5.0, 1.0), 2.0 * PI * PI * 5.0));
    }

    #[test]
    fn test_volume_frustum() {
        // When r1 == r2, frustum becomes a cylinder
        assert!(approx(volume_frustum(3.0, 3.0, 5.0), volume_cylinder(3.0, 5.0)));
        // When r2 == 0, frustum becomes a cone
        assert!(approx(volume_frustum(3.0, 0.0, 6.0), volume_cone(3.0, 6.0)));
    }

    #[test]
    fn test_volume_capsule() {
        // Capsule = cylinder + sphere
        let r = 2.0;
        let h = 5.0;
        assert!(approx(volume_capsule(r, h), volume_cylinder(r, h) + volume_sphere(r)));
    }

    // Surface Areas

    #[test]
    fn test_surface_sphere() {
        assert!(approx(surface_sphere(3.0), 4.0 * PI * 9.0));
    }

    #[test]
    fn test_surface_cylinder_total() {
        assert!(approx(surface_cylinder_total(2.0, 5.0), 2.0 * PI * 2.0 * 7.0));
    }

    #[test]
    fn test_surface_cylinder_lateral() {
        assert!(approx(surface_cylinder_lateral(2.0, 5.0), 2.0 * PI * 2.0 * 5.0));
    }

    #[test]
    fn test_surface_cone_lateral() {
        assert!(approx(surface_cone_lateral(3.0, 5.0), PI * 15.0));
    }

    #[test]
    fn test_surface_torus() {
        assert!(approx(surface_torus(5.0, 2.0), 4.0 * PI * PI * 10.0));
    }

    // Solid Angles & Spherical Geometry

    #[test]
    fn test_solid_angle_cone() {
        // Half angle = π → full sphere: 2π(1 - cos(π)) = 2π(1+1) = 4π
        assert!(approx(solid_angle_cone(PI), 4.0 * PI));
        // Half angle = 0 → zero solid angle
        assert!(approx(solid_angle_cone(0.0), 0.0));
    }

    #[test]
    fn test_solid_angle_full_sphere() {
        assert!(approx(solid_angle_full_sphere(), 4.0 * PI));
    }

    #[test]
    fn test_great_circle_distance() {
        // Same point → distance 0
        assert!(approx(great_circle_distance(6371.0, 0.5, 1.0, 0.5, 1.0), 0.0));
        // Antipodal points on unit sphere → distance = π
        assert!(approx(great_circle_distance(1.0, 0.0, 0.0, 0.0, PI), PI));
    }

    #[test]
    fn test_spherical_excess() {
        // Equilateral spherical triangle with right angles: E = 3×(π/2) - π = π/2
        let half_pi = PI / 2.0;
        assert!(approx(spherical_excess(half_pi, half_pi, half_pi), half_pi));
    }

    // Moments of Inertia

    #[test]
    fn test_moi_solid_sphere() {
        assert!(approx(moi_solid_sphere(10.0, 3.0), 2.0 * 10.0 * 9.0 / 5.0));
    }

    #[test]
    fn test_moi_hollow_sphere() {
        assert!(approx(moi_hollow_sphere(10.0, 3.0), 2.0 * 10.0 * 9.0 / 3.0));
    }

    #[test]
    fn test_moi_solid_cylinder() {
        assert!(approx(moi_solid_cylinder(8.0, 2.0), 8.0 * 4.0 / 2.0));
    }

    #[test]
    fn test_moi_thin_rod_center() {
        assert!(approx(moi_thin_rod_center(6.0, 3.0), 6.0 * 9.0 / 12.0));
    }

    #[test]
    fn test_moi_thin_rod_end() {
        assert!(approx(moi_thin_rod_end(6.0, 3.0), 6.0 * 9.0 / 3.0));
    }

    #[test]
    fn test_moi_rectangular_plate() {
        assert!(approx(moi_rectangular_plate(12.0, 3.0, 4.0), 12.0 * 25.0 / 12.0));
    }

    #[test]
    fn test_parallel_axis() {
        let i_cm = moi_solid_sphere(5.0, 2.0);
        let d = 3.0;
        assert!(approx(parallel_axis(i_cm, 5.0, d), i_cm + 5.0 * 9.0));
    }

    // Relationship tests

    #[test]
    fn test_rod_end_vs_center_parallel_axis() {
        let mass = 7.0;
        let length = 4.0;
        let i_center = moi_thin_rod_center(mass, length);
        let i_end = moi_thin_rod_end(mass, length);
        // I_end = I_center + m*(L/2)^2 via parallel axis theorem
        assert!(approx(i_end, parallel_axis(i_center, mass, length / 2.0)));
    }

    #[test]
    fn test_hollow_sphere_greater_than_solid() {
        let mass = 5.0;
        let radius = 3.0;
        assert!(moi_hollow_sphere(mass, radius) > moi_solid_sphere(mass, radius));
    }

    #[test]
    fn test_approx_rel_near_zero_b() {
        assert!(approx_rel(0.0, 0.0));
        assert!(!approx_rel(1.0, 0.0));
    }
}

pub mod delaunay;
pub mod geodesy;
pub mod hull;

pub use delaunay::{circumcircle, delaunay_2d, voronoi_cells_2d};
pub use geodesy::{
    ecef_to_enu, ecef_to_geodetic, geodetic_to_ecef, vincenty_direct, vincenty_inverse,
    Ellipsoid,
};
pub use hull::{convex_hull_2d, convex_hull_3d, point_in_polygon, polygon_area_signed};
