//! Constant-velocity (CV) Kalman filter — two states, position + velocity.
//!
//! Model (Bar-Shalom, Li & Kirubarajan, 2001, §6.2; Gelb, 1974):
//!
//! ```text
//!   state  s = [x, v]ᵀ          (position, velocity)
//!   F(dt)  = [[1, dt], [0, 1]]  (position advances by v·dt)
//!   H      = [1, 0]             (only position is measured)
//!   Q(dt)  = q · [[dt⁴/4, dt³/2],
//!                  [dt³/2, dt²  ]]   (continuous white-noise acceleration,
//!                                      q = power spectral density of the
//!                                      acceleration noise)
//! ```
//!
//! The 2×2 matrices are multiplied out by hand — no linear-algebra
//! crate is needed, and every line of the update can be checked against
//! the textbook equations. The discrete white-noise-acceleration form
//! (`Q = q·[[dt², dt],[dt, 1]]` with `q` the per-step velocity variance)
//! is also available via [`CvKalman::with_discrete_q`].
//!
//! Compared with the scalar random-walk filter ([`crate::scalar`]), the
//! CV filter tracks a *moving* target: during a ramp or a step it keeps
//! up because the velocity state carries the trend, while the scalar
//! filter always lags behind a moving signal.

use crate::errors::{require_non_negative, require_positive, Result};

/// Outcome of one predict + update step of the CV filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvStep {
    /// Predicted position / velocity before the measurement.
    pub x_pred: [f64; 2],
    /// Predicted 2×2 error covariance before the measurement.
    pub p_pred: [[f64; 2]; 2],
    /// Kalman gain `[K_x, K_v]`.
    pub gain: [f64; 2],
    /// Innovation (measurement residual) `z − x_pred`.
    pub innovation: f64,
    /// Innovation variance `H·P⁻·Hᵀ + R`.
    pub innovation_variance: f64,
    /// Updated position / velocity.
    pub x: [f64; 2],
    /// Updated 2×2 error covariance.
    pub p: [[f64; 2]; 2],
    /// Normalised innovation squared `ν² / S` (chi-square, 1 dof).
    pub nis: f64,
}

/// Constant-velocity Kalman filter (position + velocity).
#[derive(Debug, Clone, Copy)]
pub struct CvKalman {
    /// Current state `[position, velocity]`.
    pub x: [f64; 2],
    /// Current 2×2 error covariance.
    pub p: [[f64; 2]; 2],
    /// Acceleration-noise power spectral density `q` (continuous model)
    /// or per-step velocity variance (discrete model, see
    /// [`CvKalman::with_discrete_q`]).
    pub q: f64,
    /// Measurement-noise variance `R`.
    pub r: f64,
    /// `true` = continuous white-noise acceleration Q, `false` = discrete.
    pub continuous_q: bool,
}

impl CvKalman {
    /// Create a CV filter with the continuous white-noise acceleration
    /// process model (the textbook default).
    ///
    /// * `x0`, `v0` — initial position / velocity estimates.
    /// * `p0` — initial covariance (diagonal, `[[p0, 0], [0, p0]]`).
    /// * `q` — acceleration-noise PSD, `>= 0`.
    /// * `r` — measurement-noise variance, `> 0`.
    pub fn new(x0: f64, v0: f64, p0: f64, q: f64, r: f64) -> Result<Self> {
        require_non_negative(p0, "p0")?;
        require_non_negative(q, "q")?;
        require_positive(r, "r")?;
        if !x0.is_finite() || !v0.is_finite() {
            return Err(crate::errors::KalmanError::InvalidParameter(
                "x0 and v0 must be finite".to_string(),
            ));
        }
        Ok(Self {
            x: [x0, v0],
            p: [[p0, 0.0], [0.0, p0]],
            q,
            r,
            continuous_q: true,
        })
    }

    /// Create a CV filter with the discrete white-noise acceleration
    /// model: `Q = q·[[dt², dt],[dt, 1]]`, where `q` is the variance of
    /// the per-step velocity increment. Simpler to tune by hand, slightly
    /// less physical than the continuous form.
    pub fn with_discrete_q(x0: f64, v0: f64, p0: f64, q: f64, r: f64) -> Result<Self> {
        let mut filter = Self::new(x0, v0, p0, q, r)?;
        filter.continuous_q = false;
        Ok(filter)
    }

    /// Process-noise matrix for a step of length `dt`.
    pub fn process_noise(&self, dt: f64) -> [[f64; 2]; 2] {
        if self.continuous_q {
            let dt2 = dt * dt;
            let dt3 = dt2 * dt;
            let dt4 = dt3 * dt;
            let q = self.q;
            [
                [q * dt4 / 4.0, q * dt3 / 2.0],
                [q * dt3 / 2.0, q * dt2],
            ]
        } else {
            let dt2 = dt * dt;
            let q = self.q;
            [[q * dt2, q * dt], [q * dt, q]]
        }
    }

    /// Time update for a step of length `dt` (must be `> 0`).
    /// Returns `(x_pred, p_pred)`.
    pub fn predict(&mut self, dt: f64) -> Result<([f64; 2], [[f64; 2]; 2])> {
        require_positive(dt, "dt")?;
        let (x0, v0) = (self.x[0], self.x[1]);
        let (p00, p01, p11) = (self.p[0][0], self.p[0][1], self.p[1][1]);
        // F·x
        self.x = [x0 + v0 * dt, v0];
        // F·P·Fᵀ + Q, with F = [[1, dt], [0, 1]] (P is symmetric):
        //   P⁻₀₀ = P₀₀ + 2·dt·P₀₁ + dt²·P₁₁ + Q₀₀
        //   P⁻₀₁ = P₀₁ + dt·P₁₁ + Q₀₁
        //   P⁻₁₁ = P₁₁ + Q₁₁
        let qm = self.process_noise(dt);
        let p_pred = [
            [
                p00 + 2.0 * dt * p01 + dt * dt * p11 + qm[0][0],
                p01 + dt * p11 + qm[0][1],
            ],
            [
                p01 + dt * p11 + qm[0][1],
                p11 + qm[1][1],
            ],
        ];
        self.p = p_pred;
        Ok((self.x, self.p))
    }

    /// Measurement update with position measurement `z`.
    pub fn update(&mut self, z: f64) -> CvStep {
        let x_pred = self.x;
        let p_pred = self.p;
        // S = H·P⁻·Hᵀ + R = P⁻₀₀ + R  (H = [1, 0]).
        let innovation_variance = p_pred[0][0] + self.r;
        // K = P⁻·Hᵀ / S = [P⁻₀₀, P⁻₁₀] / S.
        let gain = [p_pred[0][0] / innovation_variance, p_pred[1][0] / innovation_variance];
        let innovation = z - x_pred[0];
        let x = [
            x_pred[0] + gain[0] * innovation,
            x_pred[1] + gain[1] * innovation,
        ];
        // P = (I − K·H)·P⁻ ; K·H = [[K₀, 0],[K₁, 0]]:
        //   P₀₀ = (1−K₀)·P⁻₀₀,  P₀₁ = (1−K₀)·P⁻₀₁
        //   P₁₀ = P⁻₁₀ − K₁·P⁻₀₀,  P₁₁ = P⁻₁₁ − K₁·P⁻₀₁
        // Symmetrised by construction (P₀₁ is written to both slots).
        let p = [
            [
                (1.0 - gain[0]) * p_pred[0][0],
                (1.0 - gain[0]) * p_pred[0][1],
            ],
            [
                p_pred[1][0] - gain[1] * p_pred[0][0],
                p_pred[1][1] - gain[1] * p_pred[0][1],
            ],
        ];
        self.x = x;
        self.p = p;
        let nis = innovation * innovation / innovation_variance;
        CvStep {
            x_pred,
            p_pred,
            gain,
            innovation,
            innovation_variance,
            x,
            p,
            nis,
        }
    }

    /// One full step: predict over `dt`, then update with `z`.
    pub fn step(&mut self, z: f64, dt: f64) -> Result<CvStep> {
        self.predict(dt)?;
        Ok(self.update(z))
    }

    /// Retune the noise parameters (`q >= 0`, `r > 0`).
    pub fn retune(&mut self, q: f64, r: f64) -> Result<()> {
        require_non_negative(q, "q")?;
        require_positive(r, "r")?;
        self.q = q;
        self.r = r;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference (see `examples/hand_check.md`):
    /// x=[0,0], P=I, dt=1, q=0.01 (continuous), R=0.1, z=1.0.
    /// Computed independently with Python.
    #[test]
    fn first_step_matches_hand_calculation() {
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.1).unwrap();
        let s = kf.step(1.0, 1.0).unwrap();
        assert!((s.x_pred[0] - 0.0).abs() < 1e-12);
        assert!((s.p_pred[0][0] - 2.0025).abs() < 1e-12);
        assert!((s.p_pred[0][1] - 1.005).abs() < 1e-12);
        assert!((s.p_pred[1][1] - 1.01).abs() < 1e-12);
        assert!((s.gain[0] - 0.952_437_574_3).abs() < 1e-9);
        assert!((s.gain[1] - 0.478_002_378_1).abs() < 1e-9);
        assert!((s.x[0] - 0.952_437_574_3).abs() < 1e-9);
        assert!((s.x[1] - 0.478_002_378_1).abs() < 1e-9);
        assert!((s.p[0][0] - 0.095_243_757_4).abs() < 1e-9);
        assert!((s.p[0][1] - 0.047_800_237_8).abs() < 1e-9);
        assert!((s.p[1][1] - 0.529_607_61).abs() < 1e-8);
    }

    #[test]
    fn predict_moves_position_by_velocity_times_dt() {
        let mut kf = CvKalman::new(10.0, 2.0, 1.0, 0.0, 0.1).unwrap();
        let (x, _) = kf.predict(0.5).unwrap();
        assert!((x[0] - 11.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn tracks_a_ramp_with_small_lag() {
        // True signal: x = 2·t, measurements with tiny noise.
        // After 50 steps the velocity estimate must be near 2 and the
        // position error small; a scalar random-walk filter would lag.
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.01).unwrap();
        let mut last_err = 0.0;
        for k in 1..=50 {
            let t = k as f64;
            let s = kf.step(2.0 * t, 1.0).unwrap();
            last_err = (s.x[0] - 2.0 * t).abs();
        }
        assert!(last_err < 0.5, "final position error {last_err}");
        assert!((kf.x[1] - 2.0).abs() < 0.3, "velocity {}", kf.x[1]);
    }

    #[test]
    fn covariance_stays_symmetric_and_positive() {
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.05, 0.2).unwrap();
        for k in 0..200 {
            let z = (k as f64 * 0.3).sin() + 0.1 * (k as f64).cos();
            let s = kf.step(z, 0.7).unwrap();
            assert!((s.p[0][1] - s.p[1][0]).abs() < 1e-12, "symmetry broken");
            assert!(s.p[0][0] >= 0.0 && s.p[1][1] >= 0.0, "negative variance");
            // Determinant of a covariance matrix must stay >= 0.
            let det = s.p[0][0] * s.p[1][1] - s.p[0][1] * s.p[1][0];
            assert!(det >= -1e-12, "negative determinant {det}");
        }
    }

    #[test]
    fn discrete_q_model_differs_from_continuous() {
        let mut a = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.1).unwrap();
        let mut b = CvKalman::with_discrete_q(0.0, 0.0, 1.0, 0.01, 0.1).unwrap();
        let sa = a.step(1.0, 1.0).unwrap();
        let sb = b.step(1.0, 1.0).unwrap();
        // Continuous Q(1) = 0.01·[[0.25, 0.5],[0.5, 1]]; discrete = 0.01·[[1,1],[1,1]].
        assert!((sa.p_pred[0][0] - 2.0025).abs() < 1e-12);
        assert!((sb.p_pred[0][0] - 2.01).abs() < 1e-12);
        assert!(sa.x[0] != sb.x[0]);
    }

    #[test]
    fn step_rejects_non_positive_dt() {
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.1).unwrap();
        assert!(kf.step(1.0, 0.0).is_err());
        assert!(kf.step(1.0, -1.0).is_err());
    }

    #[test]
    fn nis_is_innovation_squared_over_variance() {
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.1).unwrap();
        let s = kf.step(1.0, 1.0).unwrap();
        let expected = s.innovation * s.innovation / s.innovation_variance;
        assert!((s.nis - expected).abs() < 1e-15);
    }
}
