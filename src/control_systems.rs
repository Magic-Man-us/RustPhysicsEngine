use crate::math::constants::PI;

// ── Rise-time and settling-time constants ──────────────────────────────
const RISE_TIME_FACTOR: f64 = 2.2;
const SETTLING_TIME_FACTOR_1ST: f64 = 4.0;
const SETTLING_TIME_FACTOR_2ND: f64 = 4.0;
const SETTLING_PERCENT_SCALE: f64 = 100.0;
const DB_SCALE: f64 = 20.0;
const PHASE_OFFSET_DEG: f64 = 180.0;

// ── PID Controller ─────────────────────────────────────────────────────

pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: f64,
    prev_error: f64,
    output_min: f64,
    output_max: f64,
}

impl PidController {
    /// Create a new PID controller with gains Kp, Ki, Kd and output clamping bounds.
    pub fn new(kp: f64, ki: f64, kd: f64, output_min: f64, output_max: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            output_min,
            output_max,
        }
    }

    /// Compute PID output with anti-windup: u = Kp·e + Ki·∫e·dt + Kd·de/dt
    pub fn update(&mut self, setpoint: f64, measured: f64, dt: f64) -> f64 {
        let error = setpoint - measured;

        self.integral += error * dt;

        let derivative = if dt > 0.0 {
            (error - self.prev_error) / dt
        } else {
            0.0
        };

        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        let clamped = output.clamp(self.output_min, self.output_max);

        // Anti-windup: if output was clamped, roll back the integral contribution
        if (clamped - output).abs() > f64::EPSILON {
            self.integral -= error * dt;
        }

        self.prev_error = error;
        clamped
    }

    /// Reset the integral accumulator and previous error to zero.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

// ── Transfer Functions ─────────────────────────────────────────────────

/// First-order step response: y(t) = K(1 - e^(-t/τ))
pub fn first_order_step_response(gain: f64, tau: f64, t: f64) -> f64 {
    assert!(tau > 0.0, "time constant tau must be positive");
    gain * (1.0 - (-t / tau).exp())
}

/// First-order impulse response: h(t) = (K/τ)·e^(-t/τ)
pub fn first_order_impulse_response(gain: f64, tau: f64, t: f64) -> f64 {
    assert!(tau > 0.0, "time constant tau must be positive");
    (gain / tau) * (-t / tau).exp()
}

/// Second-order step response for underdamped, critically damped, and overdamped systems.
pub fn second_order_step_response(gain: f64, omega_n: f64, zeta: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }

    if zeta < 1.0 {
        // Underdamped
        let omega_d = omega_n * (1.0 - zeta * zeta).sqrt();
        let phi = (1.0 - zeta * zeta).sqrt().atan2(zeta);
        let envelope = (-zeta * omega_n * t).exp() / (1.0 - zeta * zeta).sqrt();
        gain * (1.0 - envelope * (omega_d * t + phi).sin())
    } else if (zeta - 1.0).abs() < f64::EPSILON {
        // Critically damped
        gain * (1.0 - (1.0 + omega_n * t) * (-omega_n * t).exp())
    } else {
        // Overdamped
        let s1 = -omega_n * (zeta - (zeta * zeta - 1.0).sqrt());
        let s2 = -omega_n * (zeta + (zeta * zeta - 1.0).sqrt());
        gain * (1.0 + (s1 * (s2 * t).exp() - s2 * (s1 * t).exp()) / (s2 - s1))
    }
}

/// Natural frequency of a second-order system: ωn = √(k/m)
pub fn second_order_natural_frequency(k: f64, m: f64) -> f64 {
    assert!(m > 0.0, "mass m must be positive");
    (k / m).sqrt()
}

/// Damping ratio of a second-order system: ζ = c/(2√(km))
pub fn second_order_damping_ratio(c: f64, k: f64, m: f64) -> f64 {
    assert!(k > 0.0, "stiffness k must be positive");
    assert!(m > 0.0, "mass m must be positive");
    c / (2.0 * (k * m).sqrt())
}

// ── System Characteristics ─────────────────────────────────────────────

/// Rise time of a first-order system (10% to 90%): tr = 2.2τ
pub fn rise_time_first_order(tau: f64) -> f64 {
    RISE_TIME_FACTOR * tau
}

/// Settling time of a first-order system (2% criterion): ts = 4τ
pub fn settling_time_first_order(tau: f64) -> f64 {
    SETTLING_TIME_FACTOR_1ST * tau
}

/// Settling time of a second-order system (2% criterion): ts = 4/(ζωn)
pub fn settling_time_second_order(zeta: f64, omega_n: f64) -> f64 {
    assert!(zeta > 0.0, "damping ratio zeta must be positive");
    assert!(omega_n > 0.0, "natural frequency omega_n must be positive");
    SETTLING_TIME_FACTOR_2ND / (zeta * omega_n)
}

/// Peak overshoot percentage: Mp = 100·exp(-πζ/√(1-ζ²))
pub fn overshoot_percent(zeta: f64) -> f64 {
    if zeta >= 1.0 {
        return 0.0;
    }
    SETTLING_PERCENT_SCALE * (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp()
}

/// Bandwidth of a first-order system: ωb = 1/τ
pub fn bandwidth_first_order(tau: f64) -> f64 {
    assert!(tau > 0.0, "time constant tau must be positive");
    1.0 / tau
}

/// Gain margin in dB: GM = -20·log₁₀(|G(jω)|) at the phase crossover frequency
pub fn gain_margin_db(open_loop_gain_at_phase_crossover: f64) -> f64 {
    assert!(open_loop_gain_at_phase_crossover != 0.0, "open_loop_gain_at_phase_crossover must be nonzero");
    -DB_SCALE * open_loop_gain_at_phase_crossover.abs().log10()
}

/// Phase margin in degrees: PM = 180° + φ(ωgc)
pub fn phase_margin(phase_at_gain_crossover: f64) -> f64 {
    PHASE_OFFSET_DEG + phase_at_gain_crossover
}

/// Steady-state error for a type-0 system with step input: ess = 1/(1 + K)
pub fn steady_state_error_type0(gain: f64) -> f64 {
    1.0 / (1.0 + gain)
}

/// Steady-state error for a type-1 system with ramp input: ess = 1/K
pub fn steady_state_error_type1(gain: f64) -> f64 {
    assert!(gain != 0.0, "gain must be nonzero");
    1.0 / gain
}

// ── Stability ──────────────────────────────────────────────────────────

/// Check stability of a first-order system: stable when τ > 0.
pub fn is_stable_first_order(tau: f64) -> bool {
    tau > 0.0
}

/// Check stability of a second-order system: stable when ζ > 0 and ωn > 0.
pub fn is_stable_second_order(zeta: f64, omega_n: f64) -> bool {
    zeta > 0.0 && omega_n > 0.0
}

/// Routh stability criterion for a 2nd-order polynomial: stable when all coefficients > 0.
pub fn routh_criterion_2nd(a0: f64, a1: f64, a2: f64) -> bool {
    a0 > 0.0 && a1 > 0.0 && a2 > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        if b.abs() < 1e-12 {
            a.abs() < tol
        } else {
            ((a - b) / b).abs() < tol
        }
    }

    #[test]
    fn pid_tracks_step_input() {
        let mut pid = PidController::new(2.0, 1.0, 0.1, -100.0, 100.0);
        let setpoint = 10.0;
        let dt = 0.01;
        let mut measured = 0.0;

        for _ in 0..5000 {
            let output = pid.update(setpoint, measured, dt);
            // Simple plant: integrator (measured += output * dt)
            measured += output * dt;
        }

        assert!(
            approx_rel(measured, setpoint, 0.01),
            "PID should track setpoint. measured={measured}, setpoint={setpoint}"
        );
    }

    #[test]
    fn first_order_step_approaches_gain() {
        let gain = 5.0;
        let tau = 1.0;

        // At t = 5τ the response should be within ~0.7% of gain
        let response = first_order_step_response(gain, tau, 5.0 * tau);
        assert!(
            approx_rel(response, gain, 0.01),
            "First-order step response at 5τ should be ~gain. got={response}"
        );

        // At t=0 should be 0
        assert!(approx(first_order_step_response(gain, tau, 0.0), 0.0));
    }

    #[test]
    fn underdamped_second_order_overshoots() {
        let gain = 1.0;
        let omega_n = 10.0;
        let zeta = 0.3;

        // Sample the response and find the peak
        let mut peak = 0.0_f64;
        let dt = 0.001;
        let mut t = 0.0;
        while t < 5.0 {
            let y = second_order_step_response(gain, omega_n, zeta, t);
            peak = peak.max(y);
            t += dt;
        }

        assert!(
            peak > gain,
            "Underdamped system must overshoot. peak={peak}, gain={gain}"
        );
    }

    #[test]
    fn critically_damped_no_overshoot() {
        let gain = 1.0;
        let omega_n = 10.0;
        let zeta = 1.0;

        let dt = 0.001;
        let mut t = 0.0;
        while t < 5.0 {
            let y = second_order_step_response(gain, omega_n, zeta, t);
            assert!(
                y <= gain + 1e-9,
                "Critically damped must not overshoot. y={y} at t={t}"
            );
            t += dt;
        }
    }

    #[test]
    fn overshoot_at_zeta_0_5() {
        let computed = overshoot_percent(0.5);
        assert!(
            approx_rel(computed, 16.303, 0.01),
            "Overshoot at zeta=0.5 should be ~16.3%. got={computed}"
        );
    }

    #[test]
    fn stability_checks() {
        assert!(is_stable_first_order(1.0));
        assert!(!is_stable_first_order(-1.0));
        assert!(is_stable_second_order(0.5, 10.0));
        assert!(!is_stable_second_order(-0.1, 10.0));
        assert!(routh_criterion_2nd(1.0, 2.0, 3.0));
        assert!(!routh_criterion_2nd(-1.0, 2.0, 3.0));
    }

    #[test]
    fn system_characteristics() {
        let tau = 0.5;
        assert!(approx(rise_time_first_order(tau), 1.1));
        assert!(approx(settling_time_first_order(tau), 2.0));
        assert!(approx(bandwidth_first_order(tau), 2.0));

        assert!(approx(settling_time_second_order(0.5, 10.0), 0.8));
        assert!(approx(steady_state_error_type0(9.0), 0.1));
        assert!(approx(steady_state_error_type1(10.0), 0.1));
    }

    #[test]
    fn gain_and_phase_margins() {
        // Gain margin: if |G| = 0.5 at phase crossover, GM = -20log10(0.5) ≈ 6.02 dB
        let gm = gain_margin_db(0.5);
        assert!(
            approx_rel(gm, 6.0206, 0.001),
            "GM mismatch: {gm}"
        );

        // Phase margin: if phase = -150° at gain crossover, PM = 180 - 150 = 30°
        let pm = phase_margin(-150.0);
        assert!(approx(pm, 30.0), "PM mismatch: {pm}");
    }

    #[test]
    fn natural_frequency_and_damping_ratio() {
        // k=100, m=1 => ωn = 10
        assert!(approx(second_order_natural_frequency(100.0, 1.0), 10.0));
        // c=10, k=100, m=1 => ζ = 10/(2*10) = 0.5
        assert!(approx(second_order_damping_ratio(10.0, 100.0, 1.0), 0.5));
    }

    #[test]
    fn first_order_impulse_response_at_zero() {
        // h(0) = K/τ
        let gain = 3.0;
        let tau = 0.5;
        let h = first_order_impulse_response(gain, tau, 0.0);
        assert!(approx(h, gain / tau), "h(0) should be K/τ = {}, got {h}", gain / tau);
    }

    #[test]
    fn first_order_impulse_response_decays() {
        let gain = 2.0;
        let tau = 1.0;
        let h_early = first_order_impulse_response(gain, tau, 0.5);
        let h_late = first_order_impulse_response(gain, tau, 5.0);
        assert!(h_early > h_late, "impulse response should decay over time");
        assert!(h_late > 0.0, "impulse response should remain positive");
    }

    #[test]
    fn pid_reset_clears_state() {
        let mut pid = PidController::new(1.0, 1.0, 0.1, -100.0, 100.0);
        // Run some updates to build up integral and prev_error
        for _ in 0..100 {
            pid.update(10.0, 0.0, 0.01);
        }
        pid.reset();
        // After reset, a zero-error update should produce zero output
        let output = pid.update(5.0, 5.0, 0.01);
        assert!(approx(output, 0.0), "after reset with zero error, output should be 0, got {output}");
    }

    #[test]
    fn pid_zero_dt() {
        let mut pid = PidController::new(1.0, 1.0, 0.1, -100.0, 100.0);
        let output = pid.update(10.0, 0.0, 0.0);
        assert!(output.is_finite());
    }

    #[test]
    fn second_order_step_response_overdamped() {
        let gain = 1.0;
        let omega_n = 10.0;
        let zeta = 2.0;
        let t = 1.0;
        let y = second_order_step_response(gain, omega_n, zeta, t);
        assert!(y > 0.0 && y <= gain, "Overdamped response should approach gain monotonically, got {y}");
    }

    #[test]
    fn second_order_step_response_at_t_zero() {
        let y = second_order_step_response(1.0, 10.0, 0.5, 0.0);
        assert!(approx(y, 0.0));
    }

    #[test]
    fn overshoot_percent_overdamped() {
        let mp = overshoot_percent(1.5);
        assert!(approx(mp, 0.0));
    }

    #[test]
    fn test_approx_rel_zero_b() {
        assert!(approx_rel(0.0, 0.0, 1e-6));
        assert!(!approx_rel(1.0, 0.0, 0.5));
    }
}
