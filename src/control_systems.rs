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

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

// ── Transfer Functions ─────────────────────────────────────────────────

pub fn first_order_step_response(gain: f64, tau: f64, t: f64) -> f64 {
    gain * (1.0 - (-t / tau).exp())
}

pub fn first_order_impulse_response(gain: f64, tau: f64, t: f64) -> f64 {
    (gain / tau) * (-t / tau).exp()
}

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

pub fn second_order_natural_frequency(k: f64, m: f64) -> f64 {
    (k / m).sqrt()
}

pub fn second_order_damping_ratio(c: f64, k: f64, m: f64) -> f64 {
    c / (2.0 * (k * m).sqrt())
}

// ── System Characteristics ─────────────────────────────────────────────

pub fn rise_time_first_order(tau: f64) -> f64 {
    RISE_TIME_FACTOR * tau
}

pub fn settling_time_first_order(tau: f64) -> f64 {
    SETTLING_TIME_FACTOR_1ST * tau
}

pub fn settling_time_second_order(zeta: f64, omega_n: f64) -> f64 {
    SETTLING_TIME_FACTOR_2ND / (zeta * omega_n)
}

pub fn overshoot_percent(zeta: f64) -> f64 {
    if zeta >= 1.0 {
        return 0.0;
    }
    SETTLING_PERCENT_SCALE * (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp()
}

pub fn bandwidth_first_order(tau: f64) -> f64 {
    1.0 / tau
}

pub fn gain_margin_db(open_loop_gain_at_phase_crossover: f64) -> f64 {
    -DB_SCALE * open_loop_gain_at_phase_crossover.abs().log10()
}

pub fn phase_margin(phase_at_gain_crossover: f64) -> f64 {
    PHASE_OFFSET_DEG + phase_at_gain_crossover
}

pub fn steady_state_error_type0(gain: f64) -> f64 {
    1.0 / (1.0 + gain)
}

pub fn steady_state_error_type1(gain: f64) -> f64 {
    1.0 / gain
}

// ── Stability ──────────────────────────────────────────────────────────

pub fn is_stable_first_order(tau: f64) -> bool {
    tau > 0.0
}

pub fn is_stable_second_order(zeta: f64, omega_n: f64) -> bool {
    zeta > 0.0 && omega_n > 0.0
}

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
        let zeta = 0.5;
        let expected = SETTLING_PERCENT_SCALE * (-PI * zeta / (1.0 - zeta * zeta).sqrt()).exp();
        let computed = overshoot_percent(zeta);

        assert!(
            approx_rel(computed, expected, 1e-9),
            "Overshoot formula mismatch. computed={computed}, expected={expected}"
        );

        // Known value: ~16.3%
        assert!(
            approx_rel(computed, 16.303, 0.01),
            "Overshoot at ζ=0.5 should be ~16.3%. got={computed}"
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
}
