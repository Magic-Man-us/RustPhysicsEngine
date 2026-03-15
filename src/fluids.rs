use crate::math::constants;

// ── Pressure ──

/// Hydrostatic pressure: P = ρ * g * h
pub fn hydrostatic_pressure(density: f64, g: f64, depth: f64) -> f64 {
    density * g * depth
}

/// Total pressure at depth: P = P_atm + ρ * g * h
pub fn total_pressure(atmospheric_pressure: f64, density: f64, g: f64, depth: f64) -> f64 {
    atmospheric_pressure + density * g * depth
}

/// Pascal's principle: F2 = F1 * (A2 / A1)
pub fn pascal_force(f1: f64, a1: f64, a2: f64) -> f64 {
    f1 * a2 / a1
}

/// Pressure: P = F / A
pub fn pressure(force: f64, area: f64) -> f64 {
    force / area
}

// ── Buoyancy ──

/// Buoyant force (Archimedes' principle): F_b = ρ_fluid * V_displaced * g
pub fn buoyant_force(fluid_density: f64, displaced_volume: f64, g: f64) -> f64 {
    fluid_density * displaced_volume * g
}

/// Fraction of object submerged (floating): f = ρ_object / ρ_fluid
pub fn fraction_submerged(object_density: f64, fluid_density: f64) -> f64 {
    object_density / fluid_density
}

/// Apparent weight in fluid: W_app = W - F_b = mg - ρ_fluid * V * g
pub fn apparent_weight(mass: f64, object_volume: f64, fluid_density: f64, g: f64) -> f64 {
    mass * g - fluid_density * object_volume * g
}

// ── Fluid Flow ──

/// Continuity equation: A1 * v1 = A2 * v2 → v2 = A1 * v1 / A2
pub fn continuity_velocity(a1: f64, v1: f64, a2: f64) -> f64 {
    a1 * v1 / a2
}

/// Volume flow rate: Q = A * v
pub fn flow_rate(area: f64, velocity: f64) -> f64 {
    area * velocity
}

/// Mass flow rate: ṁ = ρ * A * v
pub fn mass_flow_rate(density: f64, area: f64, velocity: f64) -> f64 {
    density * area * velocity
}

/// Bernoulli's equation: P1 + 0.5*ρ*v1^2 + ρ*g*h1 = P2 + 0.5*ρ*v2^2 + ρ*g*h2
/// Returns P2 given all other quantities.
pub fn bernoulli_pressure(
    p1: f64,
    density: f64,
    v1: f64,
    h1: f64,
    v2: f64,
    h2: f64,
    g: f64,
) -> f64 {
    p1 + 0.5 * density * (v1 * v1 - v2 * v2) + density * g * (h1 - h2)
}

/// Torricelli's theorem: v = sqrt(2 * g * h)
pub fn torricelli_velocity(g: f64, height: f64) -> f64 {
    (2.0 * g * height).sqrt()
}

/// Venturi effect velocity from pressure difference:
/// v2 = sqrt(2 * (P1 - P2) / (ρ * (1 - (A2/A1)^2)))
pub fn venturi_velocity(p1: f64, p2: f64, density: f64, a1: f64, a2: f64) -> f64 {
    let ratio = a2 / a1;
    (2.0 * (p1 - p2) / (density * (1.0 - ratio * ratio))).sqrt()
}

// ── Viscosity and Drag ──

/// Drag force: F_d = 0.5 * C_d * ρ * A * v^2
pub fn drag_force(drag_coefficient: f64, density: f64, area: f64, velocity: f64) -> f64 {
    0.5 * drag_coefficient * density * area * velocity * velocity
}

/// Terminal velocity: v_t = sqrt(2 * m * g / (ρ * A * C_d))
pub fn terminal_velocity(mass: f64, g: f64, density: f64, area: f64, drag_coefficient: f64) -> f64 {
    (2.0 * mass * g / (density * area * drag_coefficient)).sqrt()
}

/// Stokes' drag (low Reynolds number): F = 6π * μ * r * v
pub fn stokes_drag(dynamic_viscosity: f64, radius: f64, velocity: f64) -> f64 {
    6.0 * constants::PI * dynamic_viscosity * radius * velocity
}

/// Reynolds number: Re = ρ * v * L / μ
pub fn reynolds_number(density: f64, velocity: f64, length: f64, dynamic_viscosity: f64) -> f64 {
    density * velocity * length / dynamic_viscosity
}

/// Poiseuille's law (volume flow rate through a pipe):
/// Q = π * r^4 * ΔP / (8 * μ * L)
pub fn poiseuille_flow_rate(
    radius: f64,
    pressure_drop: f64,
    dynamic_viscosity: f64,
    length: f64,
) -> f64 {
    constants::PI * radius.powi(4) * pressure_drop / (8.0 * dynamic_viscosity * length)
}

// ── Surface Tension ──

/// Surface tension force along a line: F = γ * L
pub fn surface_tension_force(surface_tension: f64, length: f64) -> f64 {
    surface_tension * length
}

/// Capillary rise: h = 2 * γ * cos(θ) / (ρ * g * r)
pub fn capillary_rise(
    surface_tension: f64,
    contact_angle_rad: f64,
    density: f64,
    g: f64,
    tube_radius: f64,
) -> f64 {
    2.0 * surface_tension * contact_angle_rad.cos() / (density * g * tube_radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        ((a - b) / b).abs() < tol
    }

    #[test]
    fn test_hydrostatic_pressure() {
        // Water at 10m depth: P = 1000 * 9.8 * 10 = 98000 Pa
        let p = hydrostatic_pressure(1000.0, 9.8, 10.0);
        assert!(approx(p, 98000.0, 0.1));
    }

    #[test]
    fn test_buoyant_force() {
        // 0.001 m^3 in water: F = 1000 * 0.001 * 9.8 = 9.8 N
        let f = buoyant_force(1000.0, 0.001, 9.8);
        assert!(approx(f, 9.8, 1e-6));
    }

    #[test]
    fn test_bernoulli() {
        // If same height, lower velocity → higher pressure
        let p2 = bernoulli_pressure(100000.0, 1000.0, 2.0, 0.0, 4.0, 0.0, 9.8);
        assert!(p2 < 100000.0);
    }

    #[test]
    fn test_torricelli() {
        let v = torricelli_velocity(9.8, 5.0);
        assert!(approx_rel(v, 9.899, 0.01));
    }

    #[test]
    fn test_drag_force() {
        let f = drag_force(0.47, 1.225, 0.01, 10.0);
        assert!(approx_rel(f, 0.287875, 0.01));
    }

    #[test]
    fn test_reynolds_number() {
        // Water flowing at 1 m/s through 0.1m pipe, μ ≈ 0.001
        let re = reynolds_number(1000.0, 1.0, 0.1, 0.001);
        assert!(approx(re, 100000.0, 1.0));
    }

    #[test]
    fn test_continuity() {
        let v2 = continuity_velocity(0.1, 2.0, 0.05);
        assert!(approx(v2, 4.0, 1e-9));
    }

    #[test]
    fn test_pascal() {
        let f2 = pascal_force(100.0, 0.01, 1.0);
        assert!(approx(f2, 10000.0, 1e-6));
    }
}
