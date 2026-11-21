//! α–β filter — the fixed-gain cousin of the Kalman filter.
//!
//! The α–β filter (Benedict & Bordner, 1962) tracks position and
//! velocity with *constant* gains instead of the Kalman filter's
//! time-varying gain:
//!
//! ```text
//!   predict:  x⁻ = x + v·dt
//!   residual: r  = z − x⁻
//!   update:   x  = x⁻ + α·r
//!               v  = v + (β/dt)·r
//! ```
//!
//! It is the standard baseline this crate compares the Kalman filter
//! against: same two states, same measurements, but no covariance, no
//! optimality claim, and two hand-picked gains instead of a noise model.
//! For a steady-state Kalman filter there is an exact correspondence —
//! the Kalman gains converge to fixed values that an α–β filter with
//! the same gains reproduces — which makes the comparison a strong
//! cross-check of the CV implementation.
//!
//! Gains can be set directly, or derived from a **tracking index**
//! `λ = σ_w·dt² / σ_v` (Kalata, 1984), the ratio of process-noise to
//! measurement-noise scale:
//!
//! ```text
//!   β(α) = 2(2 − α) − 4√(1 − α)
//!   λ²   = β² / (α(2 − α))          (solved for α by bisection, 0 < λ < 2)
//! ```
//!
//! Small λ → small gains (smooth, slow to react); large λ → gains near
//! α=1, β=2 (reactive, noisy). λ must stay below 2, where the
//! steady-state solution saturates.

use crate::errors::{require_positive, KalmanError, Result};

/// Outcome of one α–β step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaBetaStep {
    /// Predicted position before the measurement.
    pub x_pred: f64,
    /// Residual `z − x_pred`.
    pub residual: f64,
    /// Updated position.
    pub x: f64,
    /// Updated velocity.
    pub v: f64,
}

/// α–β tracker (position + velocity, fixed gains).
#[derive(Debug, Clone, Copy)]
pub struct AlphaBeta {
    /// Current position estimate.
    pub x: f64,
    /// Current velocity estimate.
    pub v: f64,
    /// Position gain, 0 < α ≤ 1.
    pub alpha: f64,
    /// Velocity gain, 0 < β ≤ 2 (for the steady-state correspondence;
    /// the filter itself accepts any β > 0).
    pub beta: f64,
}

impl AlphaBeta {
    /// Create a tracker with explicit gains.
    /// `alpha` must be in `(0, 1]`, `beta` in `(0, 4]`.
    pub fn new(x0: f64, v0: f64, alpha: f64, beta: f64) -> Result<Self> {
        if !x0.is_finite() || !v0.is_finite() {
            return Err(KalmanError::InvalidParameter(
                "x0 and v0 must be finite".to_string(),
            ));
        }
        if !(0.0 < alpha && alpha <= 1.0) {
            return Err(KalmanError::InvalidParameter(format!(
                "alpha must be in (0, 1], got {alpha}"
            )));
        }
        if !(0.0 < beta && beta <= 4.0) {
            return Err(KalmanError::InvalidParameter(format!(
                "beta must be in (0, 4], got {beta}"
            )));
        }
        Ok(Self {
            x: x0,
            v: v0,
            alpha,
            beta,
        })
    }

    /// Create a tracker from a tracking index λ (Kalata, 1984).
    /// Requires `0 < λ < 2`; outside that range there is no steady-state
    /// α–β solution.
    pub fn from_tracking_index(x0: f64, v0: f64, lambda: f64) -> Result<Self> {
        require_positive(lambda, "lambda")?;
        if lambda >= 2.0 {
            return Err(KalmanError::InvalidParameter(format!(
                "lambda must be < 2 (steady-state saturates at 2), got {lambda}"
            )));
        }
        let (alpha, beta) = alpha_beta_from_lambda(lambda);
        Self::new(x0, v0, alpha, beta)
    }

    /// One step: predict over `dt`, then correct with measurement `z`.
    pub fn step(&mut self, z: f64, dt: f64) -> Result<AlphaBetaStep> {
        require_positive(dt, "dt")?;
        let x_pred = self.x + self.v * dt;
        let residual = z - x_pred;
        self.x = x_pred + self.alpha * residual;
        self.v += (self.beta / dt) * residual;
        Ok(AlphaBetaStep {
            x_pred,
            residual,
            x: self.x,
            v: self.v,
        })
    }
}

/// Solve Kalata's steady-state relations for a tracking index λ.
///
/// Returns `(α, β)` with `β(α) = 2(2−α) − 4√(1−α)` and
/// `λ² = β²/(α(2−α))`. Bisection on α ∈ (0, 1); the left-hand side is
/// monotone increasing in α, from 0 (α→0) to 4 (α→1).
pub fn alpha_beta_from_lambda(lambda: f64) -> (f64, f64) {
    debug_assert!(lambda > 0.0 && lambda < 2.0);
    let target = lambda * lambda;
    let mut lo = 1e-12f64;
    let mut hi = 1.0 - 1e-12f64;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if lambda_sq_of_alpha(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let alpha = (lo + hi) / 2.0;
    (alpha, beta_of_alpha(alpha))
}

/// `λ²` implied by a given α under Kalata's steady-state relations.
fn lambda_sq_of_alpha(alpha: f64) -> f64 {
    let beta = beta_of_alpha(alpha);
    beta * beta / (alpha * (2.0 - alpha))
}

/// `β(α) = 2(2 − α) − 4√(1 − α)`.
fn beta_of_alpha(alpha: f64) -> f64 {
    2.0 * (2.0 - alpha) - 4.0 * (1.0 - alpha).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_matches_hand_calculation() {
        // x=0, v=0, α=0.5, β=0.1, dt=1, z=1.0:
        // x_pred=0, r=1, x=0.5, v=0.1.
        let mut f = AlphaBeta::new(0.0, 0.0, 0.5, 0.1).unwrap();
        let s = f.step(1.0, 1.0).unwrap();
        assert!((s.x_pred - 0.0).abs() < 1e-12);
        assert!((s.residual - 1.0).abs() < 1e-12);
        assert!((s.x - 0.5).abs() < 1e-12);
        assert!((s.v - 0.1).abs() < 1e-12);
    }

    #[test]
    fn second_step_with_velocity() {
        // Continue: x=0.5, v=0.1, dt=1, z=1.2:
        // x_pred=0.6, r=0.6, x=0.9, v=0.16.
        let mut f = AlphaBeta::new(0.0, 0.0, 0.5, 0.1).unwrap();
        f.step(1.0, 1.0).unwrap();
        let s = f.step(1.2, 1.0).unwrap();
        assert!((s.x_pred - 0.6).abs() < 1e-12);
        assert!((s.residual - 0.6).abs() < 1e-12);
        assert!((s.x - 0.9).abs() < 1e-12);
        assert!((s.v - 0.16).abs() < 1e-12);
    }

    #[test]
    fn kalata_lambda_one_gives_known_gains() {
        // Solved independently with Python (bisection, 200 iterations):
        // λ=1 → α=0.913434, β=0.996246.
        let (alpha, beta) = alpha_beta_from_lambda(1.0);
        assert!((alpha - 0.913_434).abs() < 1e-5);
        assert!((beta - 0.996_246).abs() < 1e-5);
    }

    #[test]
    fn kalata_round_trip_is_consistent() {
        for lambda in [0.05, 0.2, 0.5, 1.0, 1.5, 1.9] {
            let (alpha, beta) = alpha_beta_from_lambda(lambda);
            assert!((0.0..1.0).contains(&alpha), "alpha {alpha} for λ={lambda}");
            assert!((0.0..2.0).contains(&beta), "beta {beta} for λ={lambda}");
            let back = lambda_sq_of_alpha(alpha).sqrt();
            assert!((back - lambda).abs() < 1e-9, "λ={lambda} round-trips to {back}");
        }
    }

    #[test]
    fn from_tracking_index_rejects_lambda_two_and_above() {
        assert!(AlphaBeta::from_tracking_index(0.0, 0.0, 2.0).is_err());
        assert!(AlphaBeta::from_tracking_index(0.0, 0.0, 5.0).is_err());
        assert!(AlphaBeta::from_tracking_index(0.0, 0.0, 0.0).is_err());
        assert!(AlphaBeta::from_tracking_index(0.0, 0.0, 1.0).is_ok());
    }

    #[test]
    fn constructor_rejects_bad_gains() {
        assert!(AlphaBeta::new(0.0, 0.0, 0.0, 0.1).is_err());
        assert!(AlphaBeta::new(0.0, 0.0, 1.5, 0.1).is_err());
        assert!(AlphaBeta::new(0.0, 0.0, 0.5, 0.0).is_err());
        assert!(AlphaBeta::new(0.0, 0.0, 0.5, 5.0).is_err());
    }

    #[test]
    fn tracks_a_ramp_and_converges_velocity() {
        let mut f = AlphaBeta::new(0.0, 0.0, 0.5, 0.1).unwrap();
        for k in 1..=60 {
            f.step(2.0 * k as f64, 1.0).unwrap();
        }
        assert!((f.x - 120.0).abs() < 2.0, "position {}", f.x);
        assert!((f.v - 2.0).abs() < 0.2, "velocity {}", f.v);
    }
}
