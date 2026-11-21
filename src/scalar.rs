//! Scalar (one-state) Kalman filter — the random-walk model.
//!
//! Model (Welch & Bishop, 2006, Table 1-1/1-2, reduced to one state):
//!
//! ```text
//!   x_k = x_{k-1} + w_{k-1},   w ~ N(0, Q)     (process / time update)
//!   z_k = x_k + v_k,           v ~ N(0, R)     (measurement update)
//! ```
//!
//! The filter keeps a point estimate `x` and its error variance `P`.
//! Each step is a prediction followed by a measurement update:
//!
//! ```text
//!   predict:  x⁻ = x,            P⁻ = P + Q
//!   gain:     K  = P⁻ / (P⁻ + R)
//!   update:   x  = x⁻ + K (z − x⁻),   P = (1 − K) P⁻
//! ```
//!
//! The gain is the optimal (minimum-mean-square-error) weight: when the
//! measurement is precise (`R ≪ P⁻`) the gain is near 1 and the filter
//! trusts the measurement; when the measurement is noisy (`R ≫ P⁻`) the
//! gain is near 0 and the filter trusts its prediction. The steady-state
//! gain for this model has the closed form
//! `K∞ = S / (S + R)` with `S = (Q + √(Q² + 4QR)) / 2`.
//!
//! A `step` returns a [`ScalarStep`] record so callers (CLI, smoother,
//! adaptive-noise estimator) can inspect every intermediate quantity.

use crate::errors::{require_non_negative, require_positive, Result};

/// Outcome of one predict + update step of the scalar filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarStep {
    /// Predicted (a priori) state before the measurement.
    pub x_pred: f64,
    /// Predicted (a priori) error variance before the measurement.
    pub p_pred: f64,
    /// Kalman gain used for this measurement.
    pub gain: f64,
    /// Innovation (measurement residual) `z − x_pred`.
    pub innovation: f64,
    /// Innovation variance `P⁻ + R` (used by NEES/NIS tests).
    pub innovation_variance: f64,
    /// Updated (a posteriori) state estimate.
    pub x: f64,
    /// Updated (a posteriori) error variance.
    pub p: f64,
}

/// Scalar Kalman filter for a one-dimensional random walk.
#[derive(Debug, Clone, Copy)]
pub struct ScalarKalman {
    /// Current state estimate `x̂`.
    pub x: f64,
    /// Current error variance `P`.
    pub p: f64,
    /// Process-noise variance `Q` (how much the true state wanders
    /// between steps).
    pub q: f64,
    /// Measurement-noise variance `R` (how noisy one measurement is).
    pub r: f64,
}

impl ScalarKalman {
    /// Create a filter.
    ///
    /// * `x0` — initial state estimate (use the first measurement, or 0).
    /// * `p0` — initial error variance; a large value means "I do not
    ///   trust the initial estimate" (typical: 1–1000 in the units of x²).
    /// * `q` — process-noise variance, `>= 0`.
    /// * `r` — measurement-noise variance, `> 0`.
    pub fn new(x0: f64, p0: f64, q: f64, r: f64) -> Result<Self> {
        require_non_negative(p0, "p0")?;
        require_non_negative(q, "q")?;
        require_positive(r, "r")?;
        if !x0.is_finite() {
            return Err(crate::errors::KalmanError::InvalidParameter(
                "x0 must be finite".to_string(),
            ));
        }
        Ok(Self { x: x0, p: p0, q, r })
    }

    /// Time update only: advance the state by a known control increment
    /// `u` (default 0) and grow the covariance by `Q`.
    ///
    /// Returns `(x_pred, p_pred)`.
    pub fn predict(&mut self, u: f64) -> (f64, f64) {
        self.x += u;
        self.p += self.q;
        (self.x, self.p)
    }

    /// Measurement update only, given the current prediction.
    /// Returns the [`ScalarStep`] record for this measurement.
    pub fn update(&mut self, z: f64) -> ScalarStep {
        let x_pred = self.x;
        let p_pred = self.p;
        let innovation_variance = p_pred + self.r;
        let gain = p_pred / innovation_variance;
        let innovation = z - x_pred;
        self.x = x_pred + gain * innovation;
        // Joseph form is unnecessary for a scalar; (1 − K)·P⁻ is exact
        // here and stays non-negative as long as 0 ≤ K ≤ 1.
        self.p = (1.0 - gain) * p_pred;
        ScalarStep {
            x_pred,
            p_pred,
            gain,
            innovation,
            innovation_variance,
            x: self.x,
            p: self.p,
        }
    }

    /// One full step: predict (with control increment `u`, usually 0)
    /// then update with measurement `z`.
    pub fn step(&mut self, z: f64, u: f64) -> ScalarStep {
        self.predict(u);
        self.update(z)
    }

    /// One full step with no control input (`u = 0`).
    pub fn step_plain(&mut self, z: f64) -> ScalarStep {
        self.step(z, 0.0)
    }

    /// Steady-state Kalman gain `K∞` for the current `Q` and `R`,
    /// from the closed-form solution of the scalar Riccati equation
    /// `S² − Q·S − Q·R = 0` with `S = P⁻` at steady state.
    pub fn steady_state_gain(&self) -> f64 {
        let s = (self.q + (self.q * self.q + 4.0 * self.q * self.r).sqrt()) / 2.0;
        s / (s + self.r)
    }

    /// Steady-state a posteriori error variance `P∞`.
    pub fn steady_state_variance(&self) -> f64 {
        let s = (self.q + (self.q * self.q + 4.0 * self.q * self.r).sqrt()) / 2.0;
        (1.0 - s / (s + self.r)) * s
    }

    /// Retune the noise variances (used by the adaptive estimator).
    /// `q` must be `>= 0`, `r` must be `> 0`.
    pub fn retune(&mut self, q: f64, r: f64) -> Result<()> {
        require_non_negative(q, "q")?;
        require_positive(r, "r")?;
        self.q = q;
        self.r = r;
        Ok(())
    }
}

/// Run a whole measurement series through a scalar filter, collecting
/// one [`ScalarStep`] per sample. The filter is consumed by value;
/// callers that need the final state can read it from the last step.
pub fn run_scalar_series(
    mut filter: ScalarKalman,
    measurements: &[f64],
) -> Vec<ScalarStep> {
    measurements
        .iter()
        .map(|&z| filter.step_plain(z))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference (see `examples/hand_check.md`):
    /// x0=0, P0=1, Q=0.01, R=0.1, z = [1.0, 1.1, 0.9].
    /// Computed independently with Python (double precision).
    #[test]
    fn first_three_steps_match_hand_calculation() {
        let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
        let s1 = kf.step_plain(1.0);
        assert!((s1.x_pred - 0.0).abs() < 1e-12);
        assert!((s1.p_pred - 1.01).abs() < 1e-12);
        assert!((s1.gain - 0.909_909_909_9).abs() < 1e-9);
        assert!((s1.innovation - 1.0).abs() < 1e-12);
        assert!((s1.x - 0.909_909_909_9).abs() < 1e-9);
        assert!((s1.p - 0.090_990_991_0).abs() < 1e-9);

        let s2 = kf.step_plain(1.1);
        assert!((s2.p_pred - 0.100_990_991_0).abs() < 1e-9);
        assert!((s2.gain - 0.502_465_262_2).abs() < 1e-9);
        assert!((s2.innovation - 0.190_090_090_1).abs() < 1e-9);
        assert!((s2.x - 1.005_423_576_9).abs() < 1e-9);
        assert!((s2.p - 0.050_246_526_2).abs() < 1e-9);

        let s3 = kf.step_plain(0.9);
        assert!((s3.p_pred - 0.060_246_526_2).abs() < 1e-9);
        assert!((s3.gain - 0.375_961_511_6).abs() < 1e-9);
        assert!((s3.innovation + 0.105_423_576_9).abs() < 1e-9);
        assert!((s3.x - 0.965_788_369_6).abs() < 1e-9);
        assert!((s3.p - 0.037_596_151_2).abs() < 1e-9);
    }

    #[test]
    fn gain_is_between_zero_and_one() {
        let mut kf = ScalarKalman::new(5.0, 2.0, 0.05, 0.5).unwrap();
        for z in [4.0, 6.0, 5.5, 5.0, 4.5] {
            let s = kf.step_plain(z);
            assert!((0.0..=1.0).contains(&s.gain), "gain {}", s.gain);
            assert!(s.p >= 0.0, "variance went negative: {}", s.p);
            assert!(s.p <= s.p_pred, "variance must shrink on update");
        }
    }

    #[test]
    fn steady_state_matches_iteration() {
        // Closed form vs 500 iterated steps; both must agree to 1e-12.
        let kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
        assert!((kf.steady_state_gain() - 0.270_156_211_9).abs() < 1e-9);
        assert!((kf.steady_state_variance() - 0.027_015_621_2).abs() < 1e-9);

        let mut run = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
        let mut last = ScalarStep {
            x_pred: 0.0,
            p_pred: 0.0,
            gain: 0.0,
            innovation: 0.0,
            innovation_variance: 0.0,
            x: 0.0,
            p: 0.0,
        };
        for _ in 0..500 {
            last = run.step_plain(1.0);
        }
        assert!((last.gain - kf.steady_state_gain()).abs() < 1e-9);
        assert!((last.p - kf.steady_state_variance()).abs() < 1e-9);
    }

    #[test]
    fn zero_process_noise_converges_to_mean_of_measurements() {
        // Q = 0: the filter becomes a recursive (weighted) average.
        // With P0 = 1, R = 1 the prior counts as one extra observation,
        // so after n steps x = 7·n/(n+1) and P = 1/(n+1).
        let mut kf = ScalarKalman::new(0.0, 1.0, 0.0, 1.0).unwrap();
        for _ in 0..1000 {
            kf.step_plain(7.0);
        }
        assert!((kf.x - 7.0).abs() < 0.01, "x = {}", kf.x);
        assert!(kf.p < 0.002, "variance should shrink toward 0, got {}", kf.p);
    }

    #[test]
    fn huge_measurement_noise_trusts_prediction() {
        // R >> P: gain ~ 0, estimate barely moves from the prediction.
        let mut kf = ScalarKalman::new(3.0, 0.01, 0.0, 1.0e6).unwrap();
        let s = kf.step_plain(100.0);
        assert!(s.gain < 1e-5);
        assert!((s.x - 3.0).abs() < 1e-3);
    }

    #[test]
    fn tiny_measurement_noise_trusts_measurement() {
        // R << P: gain ~ 1, estimate jumps to the measurement.
        let mut kf = ScalarKalman::new(3.0, 1.0, 0.0, 1.0e-9).unwrap();
        let s = kf.step_plain(100.0);
        assert!(s.gain > 1.0 - 1e-6);
        assert!((s.x - 100.0).abs() < 1e-6);
    }

    #[test]
    fn constructor_rejects_bad_parameters() {
        assert!(ScalarKalman::new(0.0, -1.0, 0.01, 0.1).is_err());
        assert!(ScalarKalman::new(0.0, 1.0, -0.01, 0.1).is_err());
        assert!(ScalarKalman::new(0.0, 1.0, 0.01, 0.0).is_err());
        assert!(ScalarKalman::new(f64::NAN, 1.0, 0.01, 0.1).is_err());
    }

    #[test]
    fn run_series_returns_one_step_per_sample() {
        let kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
        let steps = run_scalar_series(kf, &[1.0, 1.1, 0.9, 1.05]);
        assert_eq!(steps.len(), 4);
        assert!((steps[3].x - 0.992_944_562_0).abs() < 1e-9);
    }

    #[test]
    fn retune_changes_noise_and_rejects_bad_values() {
        let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
        kf.retune(0.05, 0.2).unwrap();
        assert_eq!(kf.q, 0.05);
        assert_eq!(kf.r, 0.2);
        assert!(kf.retune(0.01, 0.0).is_err());
        assert!(kf.retune(-0.01, 0.1).is_err());
    }
}
