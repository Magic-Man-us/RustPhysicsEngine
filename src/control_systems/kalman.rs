//! Kalman filtering.
//!
//! Linear filter: predict x ← F·x, P ← F·P·Fᵀ + Q; update with gain
//! K = P·Hᵀ·S⁻¹, S = H·P·Hᵀ + R, using the Joseph-form covariance
//! update P ← (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ so P stays symmetric PSD.
//! The gain solve goes through `linalg::lu`. Reference: Bar-Shalom,
//! Li & Kirubarajan, *Estimation with Applications to Tracking*.

use crate::error::SolveError;
use crate::linalg::{lu_decompose, Matrix};

/// Linear Kalman filter with state x, covariance P, transition F,
/// observation H, process noise Q, and measurement noise R.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanFilter {
    pub x: Vec<f64>,
    pub p: Matrix,
    pub f: Matrix,
    pub h: Matrix,
    pub q: Matrix,
    pub r: Matrix,
}

/// Shared measurement update given the innovation and Jacobian-like H.
fn joseph_update(
    x: &mut Vec<f64>,
    p: &mut Matrix,
    h: &Matrix,
    r: &Matrix,
    innovation: &[f64],
) -> Result<(), SolveError> {
    let n = x.len();
    // S = H P H^T + R.
    let hp = h.mul(p)?;
    let s = hp.mul(&h.transpose())?.add(r)?;
    // K = P H^T S^-1  ⇔  K^T = S^-1 H P (P symmetric).
    let kt = lu_decompose(&s)?.solve_matrix(&hp)?;
    let k = kt.transpose();

    let ky = k.mul_vec(innovation)?;
    for i in 0..n {
        x[i] += ky[i];
    }
    // Joseph form: P = (I - K H) P (I - K H)^T + K R K^T.
    let ikh = {
        let kh = k.mul(h)?;
        let mut m = kh.scale(-1.0);
        for i in 0..n {
            let v = m.get(i, i) + 1.0;
            m.set(i, i, v);
        }
        m
    };
    let part1 = ikh.mul(p)?.mul(&ikh.transpose())?;
    let part2 = k.mul(r)?.mul(&k.transpose())?;
    *p = part1.add(&part2)?;
    // Numerically re-symmetrize.
    let pt = p.transpose();
    *p = p.add(&pt)?.scale(0.5);
    Ok(())
}

impl KalmanFilter {
    /// Time update: x ← F·x, P ← F·P·Fᵀ + Q.
    pub fn predict(&mut self) -> Result<(), SolveError> {
        self.x = self.f.mul_vec(&self.x)?;
        self.p = self.f.mul(&self.p)?.mul(&self.f.transpose())?.add(&self.q)?;
        Ok(())
    }

    /// Measurement update with observation z.
    pub fn update(&mut self, z: &[f64]) -> Result<(), SolveError> {
        if z.len() != self.h.rows {
            return Err(SolveError::DimensionMismatch { expected: self.h.rows, got: z.len() });
        }
        let hx = self.h.mul_vec(&self.x)?;
        let innovation: Vec<f64> = z.iter().zip(&hx).map(|(zi, hi)| zi - hi).collect();
        joseph_update(&mut self.x, &mut self.p, &self.h, &self.r, &innovation)
    }

    /// 1-D constant-velocity tracker: state [position, velocity], white
    /// noise acceleration model (Q from the discretized acceleration
    /// spectral density `process_noise`).
    ///
    /// # Panics
    /// Panics unless dt, process_noise, and measurement_noise are positive.
    #[must_use]
    pub fn constant_velocity_1d(dt: f64, process_noise: f64, measurement_noise: f64) -> Self {
        assert!(dt > 0.0 && process_noise > 0.0 && measurement_noise > 0.0);
        let f = Matrix::from_rows(&[&[1.0, dt], &[0.0, 1.0]]).unwrap();
        let h = Matrix::from_rows(&[&[1.0, 0.0]]).unwrap();
        let q = Matrix::from_rows(&[
            &[dt.powi(4) / 4.0, dt.powi(3) / 2.0],
            &[dt.powi(3) / 2.0, dt * dt],
        ])
        .unwrap()
        .scale(process_noise);
        let mut r = Matrix::zeros(1, 1);
        r.set(0, 0, measurement_noise);
        Self { x: vec![0.0, 0.0], p: Matrix::identity(2).scale(100.0), f, h, q, r }
    }

    /// 3-D constant-velocity tracker: state [x, y, z, vx, vy, vz],
    /// position-only measurements.
    ///
    /// # Panics
    /// Panics unless dt, q, and r are positive.
    #[must_use]
    pub fn constant_velocity_3d(dt: f64, q: f64, r: f64) -> Self {
        assert!(dt > 0.0 && q > 0.0 && r > 0.0);
        let mut f = Matrix::identity(6);
        for i in 0..3 {
            f.set(i, i + 3, dt);
        }
        let mut h = Matrix::zeros(3, 6);
        for i in 0..3 {
            h.set(i, i, 1.0);
        }
        let mut qm = Matrix::zeros(6, 6);
        for i in 0..3 {
            qm.set(i, i, dt.powi(4) / 4.0 * q);
            qm.set(i, i + 3, dt.powi(3) / 2.0 * q);
            qm.set(i + 3, i, dt.powi(3) / 2.0 * q);
            qm.set(i + 3, i + 3, dt * dt * q);
        }
        let rm = Matrix::identity(3).scale(r);
        Self { x: vec![0.0; 6], p: Matrix::identity(6).scale(100.0), f, h, q: qm, r: rm }
    }
}

/// Extended Kalman filter: nonlinear transition f and observation h
/// with user-supplied Jacobians, linearized at the current estimate.
#[allow(clippy::type_complexity)]
pub struct ExtendedKalmanFilter {
    pub x: Vec<f64>,
    pub p: Matrix,
    pub f: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    pub jac_f: Box<dyn Fn(&[f64]) -> Matrix>,
    pub h: Box<dyn Fn(&[f64]) -> Vec<f64>>,
    pub jac_h: Box<dyn Fn(&[f64]) -> Matrix>,
    pub q: Matrix,
    pub r: Matrix,
}

impl ExtendedKalmanFilter {
    /// Time update: x ← f(x), P ← J_f·P·J_fᵀ + Q.
    pub fn predict(&mut self) -> Result<(), SolveError> {
        let jf = (self.jac_f)(&self.x);
        self.x = (self.f)(&self.x);
        if self.x.len() != jf.rows {
            return Err(SolveError::DimensionMismatch { expected: jf.rows, got: self.x.len() });
        }
        self.p = jf.mul(&self.p)?.mul(&jf.transpose())?.add(&self.q)?;
        Ok(())
    }

    /// Measurement update with observation z, linearizing h at x.
    pub fn update(&mut self, z: &[f64]) -> Result<(), SolveError> {
        let jh = (self.jac_h)(&self.x);
        if z.len() != jh.rows {
            return Err(SolveError::DimensionMismatch { expected: jh.rows, got: z.len() });
        }
        let hx = (self.h)(&self.x);
        let innovation: Vec<f64> = z.iter().zip(&hx).map(|(zi, hi)| zi - hi).collect();
        joseph_update(&mut self.x, &mut self.p, &jh, &self.r, &innovation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::eigen_symmetric;
    use crate::monte_carlo::Rng;

    #[test]
    fn test_filter_beats_raw_measurements() {
        let mut rng = Rng::new(91);
        let dt = 0.1;
        let meas_sigma = 2.0;
        let mut kf = KalmanFilter::constant_velocity_1d(dt, 0.05, meas_sigma * meas_sigma);
        let mut true_pos = 0.0;
        let true_vel = 1.5;
        let mut err_filter = 0.0;
        let mut err_raw = 0.0;
        let steps = 400;
        for _ in 0..steps {
            true_pos += true_vel * dt;
            let z = true_pos + meas_sigma * rng.next_gaussian();
            kf.predict().unwrap();
            kf.update(&[z]).unwrap();
            err_filter += (kf.x[0] - true_pos).powi(2);
            err_raw += (z - true_pos).powi(2);
        }
        let rmse_filter = (err_filter / steps as f64).sqrt();
        let rmse_raw = (err_raw / steps as f64).sqrt();
        assert!(
            rmse_filter < 0.6 * rmse_raw,
            "filter RMSE {rmse_filter} vs raw {rmse_raw}"
        );
        // Velocity estimate should approach the truth.
        assert!((kf.x[1] - true_vel).abs() < 0.5, "vel = {}", kf.x[1]);
    }

    #[test]
    fn test_covariance_stays_symmetric_psd() {
        let mut rng = Rng::new(92);
        let mut kf = KalmanFilter::constant_velocity_1d(0.1, 0.05, 1.0);
        for k in 0..200 {
            kf.predict().unwrap();
            kf.update(&[k as f64 * 0.1 + rng.next_gaussian()]).unwrap();
            assert!(kf.p.is_symmetric(1e-9), "P asymmetric at step {k}");
            let eig = eigen_symmetric(&kf.p, 1e-12, 100).unwrap();
            assert!(
                eig.values.iter().all(|&v| v > -1e-9),
                "P not PSD at step {k}: {:?}",
                eig.values
            );
        }
    }

    #[test]
    fn test_constant_velocity_3d_tracks() {
        let mut rng = Rng::new(93);
        let dt = 0.1;
        let mut kf = KalmanFilter::constant_velocity_3d(dt, 0.05, 1.0);
        let vel = [1.0, -0.5, 0.25];
        let mut pos = [0.0; 3];
        for _ in 0..300 {
            for i in 0..3 {
                pos[i] += vel[i] * dt;
            }
            let z: Vec<f64> = pos.iter().map(|p| p + rng.next_gaussian()).collect();
            kf.predict().unwrap();
            kf.update(&z).unwrap();
        }
        for i in 0..3 {
            assert!((kf.x[i] - pos[i]).abs() < 1.0, "pos {i}: {} vs {}", kf.x[i], pos[i]);
            assert!((kf.x[i + 3] - vel[i]).abs() < 0.3, "vel {i}: {}", kf.x[i + 3]);
        }
    }

    #[test]
    fn test_update_dimension_mismatch() {
        let mut kf = KalmanFilter::constant_velocity_1d(0.1, 0.1, 1.0);
        assert!(matches!(
            kf.update(&[1.0, 2.0]),
            Err(SolveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_ekf_linear_case_matches_kf() {
        // With linear f and h, the EKF reduces exactly to the KF.
        let dt = 0.1;
        let mut kf = KalmanFilter::constant_velocity_1d(dt, 0.05, 1.0);
        let f_mat = kf.f.clone();
        let h_mat = kf.h.clone();
        let mut ekf = ExtendedKalmanFilter {
            x: kf.x.clone(),
            p: kf.p.clone(),
            f: {
                let fm = f_mat.clone();
                Box::new(move |x: &[f64]| fm.mul_vec(x).unwrap())
            },
            jac_f: {
                let fm = f_mat.clone();
                Box::new(move |_: &[f64]| fm.clone())
            },
            h: {
                let hm = h_mat.clone();
                Box::new(move |x: &[f64]| hm.mul_vec(x).unwrap())
            },
            jac_h: {
                let hm = h_mat.clone();
                Box::new(move |_: &[f64]| hm.clone())
            },
            q: kf.q.clone(),
            r: kf.r.clone(),
        };
        for k in 0..50 {
            let z = [k as f64 * 0.2];
            kf.predict().unwrap();
            kf.update(&z).unwrap();
            ekf.predict().unwrap();
            ekf.update(&z).unwrap();
        }
        for (a, b) in kf.x.iter().zip(&ekf.x) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
