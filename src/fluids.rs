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
    assert!(a1 > 0.0, "a1 must be positive");
    f1 * a2 / a1
}

/// Pressure: P = F / A
pub fn pressure(force: f64, area: f64) -> f64 {
    assert!(area > 0.0, "area must be positive");
    force / area
}

// ── Buoyancy ──

/// Buoyant force (Archimedes' principle): F_b = ρ_fluid * V_displaced * g
pub fn buoyant_force(fluid_density: f64, displaced_volume: f64, g: f64) -> f64 {
    fluid_density * displaced_volume * g
}

/// Fraction of object submerged (floating): f = ρ_object / ρ_fluid
pub fn fraction_submerged(object_density: f64, fluid_density: f64) -> f64 {
    assert!(fluid_density > 0.0, "fluid_density must be positive");
    object_density / fluid_density
}

/// Apparent weight in fluid: W_app = W - F_b = mg - ρ_fluid * V * g
pub fn apparent_weight(mass: f64, object_volume: f64, fluid_density: f64, g: f64) -> f64 {
    mass * g - fluid_density * object_volume * g
}

// ── Fluid Flow ──

/// Continuity equation: A1 * v1 = A2 * v2 → v2 = A1 * v1 / A2
pub fn continuity_velocity(a1: f64, v1: f64, a2: f64) -> f64 {
    assert!(a2 > 0.0, "a2 must be positive");
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
    assert!(density > 0.0, "density must be positive");
    assert!(a1 > 0.0, "a1 must be positive");
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
    assert!(density > 0.0, "density must be positive");
    assert!(area > 0.0, "area must be positive");
    assert!(drag_coefficient > 0.0, "drag_coefficient must be positive");
    (2.0 * mass * g / (density * area * drag_coefficient)).sqrt()
}

/// Stokes' drag (low Reynolds number): F = 6π * μ * r * v
pub fn stokes_drag(dynamic_viscosity: f64, radius: f64, velocity: f64) -> f64 {
    6.0 * constants::PI * dynamic_viscosity * radius * velocity
}

/// Reynolds number: Re = ρ * v * L / μ
pub fn reynolds_number(density: f64, velocity: f64, length: f64, dynamic_viscosity: f64) -> f64 {
    assert!(dynamic_viscosity > 0.0, "dynamic_viscosity must be positive");
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
    assert!(dynamic_viscosity > 0.0, "dynamic_viscosity must be positive");
    assert!(length > 0.0, "length must be positive");
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
    assert!(density > 0.0, "density must be positive");
    assert!(g > 0.0, "g must be positive");
    assert!(tube_radius > 0.0, "tube_radius must be positive");
    2.0 * surface_tension * contact_angle_rad.cos() / (density * g * tube_radius)
}

// ── Compressible Flow ──

/// Mach number: M = v / a
pub fn mach_number(velocity: f64, speed_of_sound: f64) -> f64 {
    assert!(speed_of_sound > 0.0, "speed_of_sound must be positive");
    velocity / speed_of_sound
}

/// Dynamic pressure: q = ½ρv²
pub fn dynamic_pressure(density: f64, velocity: f64) -> f64 {
    0.5 * density * velocity * velocity
}

/// Stagnation pressure: P₀ = P + q
pub fn stagnation_pressure(static_pressure: f64, dynamic_pressure: f64) -> f64 {
    static_pressure + dynamic_pressure
}

/// Isentropic pressure ratio: P/P₀ = (1 + (γ-1)/2 × M²)^(-γ/(γ-1))
pub fn isentropic_pressure_ratio(mach: f64, gamma: f64) -> f64 {
    assert!(gamma != 1.0, "gamma must not equal 1.0");
    let exponent = -gamma / (gamma - 1.0);
    (1.0 + (gamma - 1.0) / 2.0 * mach * mach).powf(exponent)
}

/// Isentropic temperature ratio: T/T₀ = (1 + (γ-1)/2 × M²)^(-1)
pub fn isentropic_temperature_ratio(mach: f64, gamma: f64) -> f64 {
    1.0 / (1.0 + (gamma - 1.0) / 2.0 * mach * mach)
}

// ── Vorticity & Circulation ──

/// Vorticity in 2D: ω = ∂v_y/∂x - ∂v_x/∂y
pub fn vorticity_2d(dvx_dy: f64, dvy_dx: f64) -> f64 {
    dvy_dx - dvx_dy
}

/// Circulation (uniform vorticity): Γ = ω × A
pub fn circulation(vorticity: f64, area: f64) -> f64 {
    vorticity * area
}

/// Kutta-Joukowski lift per unit span: L = ρ × V × Γ
pub fn kutta_joukowski_lift(density: f64, velocity: f64, circulation: f64) -> f64 {
    density * velocity * circulation
}

// ── Navier-Stokes helpers ──

/// Kinematic viscosity: ν = μ/ρ
pub fn kinematic_viscosity(dynamic_viscosity: f64, density: f64) -> f64 {
    assert!(density > 0.0, "density must be positive");
    dynamic_viscosity / density
}

/// Pressure gradient in a pipe (Poiseuille inverse): dP = 8μLQ/(πr⁴)
pub fn pressure_gradient_pipe(
    flow_rate: f64,
    dynamic_viscosity: f64,
    radius: f64,
    length: f64,
) -> f64 {
    assert!(radius > 0.0, "radius must be positive");
    8.0 * dynamic_viscosity * length * flow_rate / (constants::PI * radius.powi(4))
}

/// Hydraulic diameter: D_h = 4A/P
pub fn hydraulic_diameter(area: f64, wetted_perimeter: f64) -> f64 {
    assert!(wetted_perimeter > 0.0, "wetted_perimeter must be positive");
    4.0 * area / wetted_perimeter
}

/// Darcy friction factor for laminar pipe flow: f = 64/Re
pub fn darcy_friction_factor_laminar(reynolds: f64) -> f64 {
    assert!(reynolds > 0.0, "reynolds must be positive");
    64.0 / reynolds
}

/// Darcy-Weisbach head loss: h_L = f × (L/D) × v²/(2g)
pub fn darcy_weisbach_head_loss(
    friction_factor: f64,
    length: f64,
    diameter: f64,
    velocity: f64,
    g: f64,
) -> f64 {
    assert!(diameter > 0.0, "diameter must be positive");
    assert!(g > 0.0, "g must be positive");
    friction_factor * (length / diameter) * velocity * velocity / (2.0 * g)
}

// ── Buoyancy-driven flow (natural convection coupling) ──

/// Characteristic buoyancy velocity: v = √(gβΔTL)
pub fn buoyancy_velocity(g: f64, beta: f64, delta_temp: f64, length: f64) -> f64 {
    (g * beta * delta_temp * length).sqrt()
}

/// Thermal expansion coefficient for ideal gas: β = 1/T
pub fn thermal_expansion_coefficient_ideal_gas(temperature: f64) -> f64 {
    assert!(temperature > 0.0, "temperature must be positive");
    1.0 / temperature
}

// ── Thermal-Fluid Coupling ──

/// Sutherland's law for viscosity: μ = μ₀ × (T/T₀)^(3/2) × (T₀ + S)/(T + S)
pub fn viscosity_sutherland(mu0: f64, t0: f64, t: f64, s: f64) -> f64 {
    assert!(t0 > 0.0, "t0 must be positive");
    assert!(t + s != 0.0, "t + s must not be zero");
    mu0 * (t / t0).powf(1.5) * (t0 + s) / (t + s)
}

/// Sutherland's law for thermal conductivity (same form as viscosity)
pub fn thermal_conductivity_gas(k0: f64, t0: f64, t: f64, s: f64) -> f64 {
    assert!(t0 > 0.0, "t0 must be positive");
    assert!(t + s != 0.0, "t + s must not be zero");
    k0 * (t / t0).powf(1.5) * (t0 + s) / (t + s)
}

/// Churchill-Chu correlation for natural convection on a vertical plate (air, Pr≈0.71):
/// Nu = (0.825 + 0.387 × Ra^(1/6) / 1.1936)²
pub fn natural_convection_nu_vertical(rayleigh: f64) -> f64 {
    let term = 0.825 + 0.387 * rayleigh.powf(1.0 / 6.0) / 1.1936;
    term * term
}

/// Natural convection Nusselt number for hot horizontal plate facing up:
/// Nu = 0.54 × Ra^(1/4), valid for 10⁴ ≤ Ra ≤ 10⁷
pub fn natural_convection_nu_horizontal_hot(rayleigh: f64) -> f64 {
    0.54 * rayleigh.powf(0.25)
}

/// Marangoni number: Ma = -(dσ/dT) × L × ΔT / (μ × α)
pub fn marangoni_number(
    surface_tension_gradient: f64,
    length: f64,
    delta_temp: f64,
    dynamic_viscosity: f64,
    thermal_diffusivity: f64,
) -> f64 {
    assert!(dynamic_viscosity > 0.0, "dynamic_viscosity must be positive");
    assert!(thermal_diffusivity > 0.0, "thermal_diffusivity must be positive");
    -surface_tension_gradient * length * delta_temp
        / (dynamic_viscosity * thermal_diffusivity)
}

/// Bond number: Bo = Δρ × g × L² / σ (gravity vs surface tension)
pub fn bond_number(density_diff: f64, g: f64, length: f64, surface_tension: f64) -> f64 {
    assert!(surface_tension > 0.0, "surface_tension must be positive");
    density_diff * g * length * length / surface_tension
}

/// Weber number: We = ρv²L / σ (inertia vs surface tension)
pub fn weber_number(density: f64, velocity: f64, length: f64, surface_tension: f64) -> f64 {
    assert!(surface_tension > 0.0, "surface_tension must be positive");
    density * velocity * velocity * length / surface_tension
}

/// Froude number: Fr = v / √(gL) (inertia vs gravity in free surface flow)
pub fn froude_number(velocity: f64, g: f64, length: f64) -> f64 {
    assert!(g > 0.0, "g must be positive");
    assert!(length > 0.0, "length must be positive");
    velocity / (g * length).sqrt()
}

/// Archimedes number: Ar = g × d³ × ρf × (ρp - ρf) / μ²
pub fn archimedes_number(
    density_fluid: f64,
    density_particle: f64,
    diameter: f64,
    dynamic_viscosity: f64,
    g: f64,
) -> f64 {
    assert!(dynamic_viscosity != 0.0, "dynamic_viscosity must not be zero");
    g * diameter.powi(3) * density_fluid * (density_particle - density_fluid)
        / (dynamic_viscosity * dynamic_viscosity)
}

/// Peclet number: Pe = vL/α (advection vs diffusion)
pub fn peclet_number(velocity: f64, length: f64, diffusivity: f64) -> f64 {
    assert!(diffusivity > 0.0, "diffusivity must be positive");
    velocity * length / diffusivity
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

    // ── Compressible Flow ──

    #[test]
    fn test_mach_number() {
        // 340 m/s in air (speed of sound ≈ 343 m/s)
        let m = mach_number(340.0, 343.0);
        assert!(approx_rel(m, 0.99125, 0.01));
    }

    #[test]
    fn test_dynamic_pressure() {
        // Air at sea level (ρ=1.225), 50 m/s: q = 0.5*1.225*2500 = 1531.25
        let q = dynamic_pressure(1.225, 50.0);
        assert!(approx(q, 1531.25, 0.01));
    }

    #[test]
    fn test_stagnation_pressure() {
        let p0 = stagnation_pressure(101325.0, 1531.25);
        assert!(approx(p0, 102856.25, 0.01));
    }

    #[test]
    fn test_isentropic_pressure_ratio() {
        // M=0.5, γ=1.4: ratio = (1 + 0.2*0.25)^(-3.5) = 1.05^(-3.5) ≈ 0.8430
        let ratio = isentropic_pressure_ratio(0.5, 1.4);
        assert!(approx_rel(ratio, 0.8430, 0.01));
    }

    #[test]
    fn test_isentropic_temperature_ratio() {
        // M=0.5, γ=1.4: T/T₀ = 1/(1 + 0.2*0.25) = 1/1.05 ≈ 0.95238
        let ratio = isentropic_temperature_ratio(0.5, 1.4);
        assert!(approx_rel(ratio, 0.95238, 0.001));
    }

    // ── Vorticity & Circulation ──

    #[test]
    fn test_vorticity_2d() {
        // ∂v_y/∂x = 3.0, ∂v_x/∂y = 1.0 → ω = 3.0 - 1.0 = 2.0
        let omega = vorticity_2d(1.0, 3.0);
        assert!(approx(omega, 2.0, 1e-9));
    }

    #[test]
    fn test_circulation() {
        let gamma = circulation(2.0, 5.0);
        assert!(approx(gamma, 10.0, 1e-9));
    }

    #[test]
    fn test_kutta_joukowski_lift() {
        // ρ=1.225, V=50, Γ=10 → L = 612.5
        let lift = kutta_joukowski_lift(1.225, 50.0, 10.0);
        assert!(approx(lift, 612.5, 0.01));
    }

    // ── Navier-Stokes helpers ──

    #[test]
    fn test_kinematic_viscosity() {
        // Water: μ=0.001 Pa·s, ρ=1000 → ν = 1e-6 m²/s
        let nu = kinematic_viscosity(0.001, 1000.0);
        assert!(approx(nu, 1e-6, 1e-12));
    }

    #[test]
    fn test_pressure_gradient_pipe() {
        // Round-trip with poiseuille_flow_rate
        let radius = 0.01;
        let mu = 0.001;
        let length = 1.0;
        let dp = 1000.0;
        let q = poiseuille_flow_rate(radius, dp, mu, length);
        let dp_back = pressure_gradient_pipe(q, mu, radius, length);
        assert!(approx_rel(dp_back, dp, 1e-9));
    }

    #[test]
    fn test_hydraulic_diameter() {
        // Square duct 0.1×0.1: A=0.01, P=0.4 → D_h = 4*0.01/0.4 = 0.1
        let dh = hydraulic_diameter(0.01, 0.4);
        assert!(approx(dh, 0.1, 1e-9));
    }

    #[test]
    fn test_darcy_friction_factor_laminar() {
        // Re=2000 → f = 64/2000 = 0.032
        let f = darcy_friction_factor_laminar(2000.0);
        assert!(approx(f, 0.032, 1e-9));
    }

    #[test]
    fn test_darcy_weisbach_head_loss() {
        // f=0.032, L=10, D=0.1, v=2, g=9.81
        // h_L = 0.032 * (10/0.1) * 4/(2*9.81) = 0.032 * 100 * 0.20387 ≈ 0.6524
        let hl = darcy_weisbach_head_loss(0.032, 10.0, 0.1, 2.0, 9.81);
        assert!(approx_rel(hl, 0.6524, 0.01));
    }

    // ── Buoyancy-driven flow ──

    #[test]
    fn test_buoyancy_velocity() {
        // g=9.81, β=0.003, ΔT=20, L=1 → v = √(9.81*0.003*20*1) = √0.5886 ≈ 0.7673
        let v = buoyancy_velocity(9.81, 0.003, 20.0, 1.0);
        assert!(approx_rel(v, 0.7673, 0.01));
    }

    #[test]
    fn test_thermal_expansion_coefficient_ideal_gas() {
        // T=300K → β = 1/300 ≈ 0.003333
        let beta = thermal_expansion_coefficient_ideal_gas(300.0);
        assert!(approx_rel(beta, 0.003333, 0.01));
    }

    // ── Thermal-Fluid Coupling ──

    #[test]
    fn test_viscosity_sutherland() {
        // Air at 400K: μ₀=1.716e-5, T₀=273.15, S=110.4
        // μ = 1.716e-5 × (400/273.15)^1.5 × (273.15+110.4)/(400+110.4)
        //   = 1.716e-5 × 1.7614 × 383.55/510.4 ≈ 2.272e-5
        let mu = viscosity_sutherland(1.716e-5, 273.15, 400.0, 110.4);
        assert!(approx_rel(mu, 2.272e-5, 0.01));
    }

    #[test]
    fn test_thermal_conductivity_gas() {
        // Air: k₀=0.0241 W/(m·K) at T₀=273.15K, S=194.0
        // At 500K: k = 0.0241 × (500/273.15)^1.5 × (273.15+194)/(500+194)
        let k = thermal_conductivity_gas(0.0241, 273.15, 500.0, 194.0);
        assert!(approx_rel(k, 0.04018, 0.02));
    }

    #[test]
    fn test_natural_convection_nu_vertical() {
        // Ra=1e9: Nu = (0.825 + 0.387*1e9^(1/6)/1.1936)²
        // 1e9^(1/6) ≈ 31.623, 0.387*31.623/1.1936 ≈ 10.247
        // Nu ≈ (0.825 + 10.247)² ≈ 122.45
        let nu = natural_convection_nu_vertical(1e9);
        assert!(approx_rel(nu, 122.45, 0.02));
    }

    #[test]
    fn test_natural_convection_nu_horizontal_hot() {
        // Ra=1e6: Nu = 0.54 × (1e6)^0.25 = 0.54 × 31.623 ≈ 17.076
        let nu = natural_convection_nu_horizontal_hot(1e6);
        assert!(approx_rel(nu, 17.076, 0.01));
    }

    #[test]
    fn test_marangoni_number() {
        // dσ/dT = -1.5e-4 N/(m·K), L=0.01m, ΔT=10K, μ=0.001 Pa·s, α=1.5e-7 m²/s
        // Ma = -(-1.5e-4) × 0.01 × 10 / (0.001 × 1.5e-7) = 1.5e-5 / 1.5e-10 = 100000
        let ma = marangoni_number(-1.5e-4, 0.01, 10.0, 0.001, 1.5e-7);
        assert!(approx_rel(ma, 100000.0, 0.01));
    }

    #[test]
    fn test_bond_number() {
        // Water-air: Δρ=1000, g=9.81, L=0.001m, σ=0.0728
        // Bo = 1000 × 9.81 × 1e-6 / 0.0728 ≈ 0.1348
        let bo = bond_number(1000.0, 9.81, 0.001, 0.0728);
        assert!(approx_rel(bo, 0.1348, 0.01));
    }

    #[test]
    fn test_weber_number() {
        // Water droplet: ρ=1000, v=10, L=0.002, σ=0.0728
        // We = 1000 × 100 × 0.002 / 0.0728 ≈ 2747.3
        let we = weber_number(1000.0, 10.0, 0.002, 0.0728);
        assert!(approx_rel(we, 2747.3, 0.01));
    }

    #[test]
    fn test_froude_number() {
        // v=3 m/s, g=9.81, L=1m: Fr = 3/√9.81 ≈ 0.9582
        let fr = froude_number(3.0, 9.81, 1.0);
        assert!(approx_rel(fr, 0.9582, 0.01));
    }

    #[test]
    fn test_archimedes_number() {
        // Sand in water: ρf=1000, ρp=2650, d=0.001, μ=0.001, g=9.81
        // Ar = 9.81 × 1e-9 × 1000 × 1650 / 1e-6 = 16178.5
        let ar = archimedes_number(1000.0, 2650.0, 0.001, 0.001, 9.81);
        assert!(approx_rel(ar, 16178.5, 0.01));
    }

    #[test]
    fn test_peclet_number() {
        // v=0.1 m/s, L=0.01m, α=1.43e-7 m²/s (water thermal diffusivity)
        // Pe = 0.1 × 0.01 / 1.43e-7 ≈ 6993.0
        let pe = peclet_number(0.1, 0.01, 1.43e-7);
        assert!(approx_rel(pe, 6993.0, 0.01));
    }

    // ── Pressure (untested) ──

    #[test]
    fn test_total_pressure() {
        // P_atm=101325 Pa, water at 10m: P = 101325 + 1000*9.8*10 = 199325 Pa
        let p = total_pressure(101325.0, 1000.0, 9.8, 10.0);
        assert!(approx(p, 199325.0, 0.1));
    }

    #[test]
    fn test_pressure() {
        // F=500 N, A=0.1 m² → P = 5000 Pa
        let p = pressure(500.0, 0.1);
        assert!(approx(p, 5000.0, 1e-9));
    }

    // ── Buoyancy (untested) ──

    #[test]
    fn test_fraction_submerged() {
        // Wood (ρ=600) in water (ρ=1000): fraction = 0.6
        let f = fraction_submerged(600.0, 1000.0);
        assert!(approx(f, 0.6, 1e-9));
    }

    #[test]
    fn test_apparent_weight() {
        // 5 kg steel (V=0.000633 m³) in water: W_app = 5*9.8 - 1000*0.000633*9.8
        let w = apparent_weight(5.0, 0.000633, 1000.0, 9.8);
        // W_app = 5*9.8 - 1000*0.000633*9.8 = 49 - 6.2034 = 42.7966 N
        assert!(approx(w, 42.7966, 1e-3));
    }

    // ── Fluid Flow (untested) ──

    #[test]
    fn test_flow_rate() {
        // A=0.01 m², v=3.0 m/s → Q = 0.03 m³/s
        let q = flow_rate(0.01, 3.0);
        assert!(approx(q, 0.03, 1e-9));
    }

    #[test]
    fn test_mass_flow_rate() {
        // ρ=1000, A=0.01, v=3.0 → ṁ = 30 kg/s
        let m = mass_flow_rate(1000.0, 0.01, 3.0);
        assert!(approx(m, 30.0, 1e-9));
    }

    #[test]
    fn test_venturi_velocity() {
        // P1=200000 Pa, P2=180000 Pa, ρ=1000, A1=0.1, A2=0.05
        // v2 = sqrt(2*20000 / (1000*(1 - 0.25))) = sqrt(40000/750) = sqrt(53.333) ≈ 7.303
        let v = venturi_velocity(200000.0, 180000.0, 1000.0, 0.1, 0.05);
        // v2 = sqrt(2*20000 / (1000*0.75)) = sqrt(53.333) ≈ 7.303 m/s
        assert!(approx_rel(v, 7.303, 0.001));
    }

    // ── Viscosity and Drag (untested) ──

    #[test]
    fn test_terminal_velocity() {
        // m=0.01 kg, g=9.8, ρ=1.225, A=0.001, Cd=0.47
        // v_t = sqrt(2*0.01*9.8 / (1.225*0.001*0.47))
        let v = terminal_velocity(0.01, 9.8, 1.225, 0.001, 0.47);
        // v_t = sqrt(2*0.01*9.8 / (1.225*0.001*0.47)) ≈ 18.449 m/s
        assert!(approx_rel(v, 18.449, 0.001));
    }

    #[test]
    fn test_stokes_drag() {
        // μ=0.001, r=0.001, v=0.01 → F = 6π*0.001*0.001*0.01
        let f = stokes_drag(0.001, 0.001, 0.01);
        // F = 6π × 0.001 × 0.001 × 0.01 ≈ 1.884956e-7 N
        assert!(approx_rel(f, 1.884956e-7, 1e-4));
    }

    // ── Surface Tension (untested) ──

    #[test]
    fn test_surface_tension_force() {
        // γ=0.0728 N/m, L=0.1 m → F = 0.00728 N
        let f = surface_tension_force(0.0728, 0.1);
        assert!(approx(f, 0.00728, 1e-9));
    }

    #[test]
    fn test_capillary_rise() {
        // Water in glass: γ=0.0728, θ=0 (cos=1), ρ=1000, g=9.8, r=0.0005
        // h = 2*0.0728*1 / (1000*9.8*0.0005) = 0.1456/4.9 ≈ 0.02971 m
        let h = capillary_rise(0.0728, 0.0, 1000.0, 9.8, 0.0005);
        // h = 2*0.0728 / (1000*9.8*0.0005) = 0.1456/4.9 ≈ 0.029714 m
        assert!(approx_rel(h, 0.029714, 0.001));
    }
}
