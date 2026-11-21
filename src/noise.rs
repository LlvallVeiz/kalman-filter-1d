//! Adaptive noise estimation — tuning Q and R from the innovations.
//!
//! The Kalman filter is only optimal when `Q` and `R` match the real
//! process and measurement noise. In practice they are guessed, so this
//! module estimates them from the data itself, following the
//! innovation-based covariance-matching idea of Mehra (1970):
//!
//! The innovation `ν_k = z_k − x̂⁻_k` of a *correctly tuned* filter is
//! white with variance `S_k = P⁻_k + R`. If the filter is mis-tuned the
//! observed innovation variance disagrees with `S_k`, and the mismatch
//! tells us how to correct `R`:
//!
//! ```text
//!   R̂ = mean(ν²) − mean(P⁻)        (covariance matching)
//! ```
//!
//! `Q` is adapted indirectly: the same mismatch, expressed through the
//! normalised innovation squared (NIS), scales `Q` up when the filter
//! is over-confident (mean NIS > 1, the state is moving faster than the
//! model allows) and down when it is under-confident. Both estimates
//! use an exponentially weighted moving average (forgetting factor λ ∈
//! (0, 1]) so the filter can follow slowly drifting noise levels.
//!
//! These are practical, widely used heuristics — not a proof of
//! optimality. The module reports every estimate it makes so the user
//! can see whether adaptation converged or chased noise.

use crate::errors::Result;

/// Running innovation statistics with exponential forgetting.
///
/// Keeps `mean(ν²)`, `mean(P⁻)` and `mean(NIS)` over the processed
/// innovations. `lambda = 1` gives the plain running mean (every sample
/// weighted equally); smaller values forget the past faster
/// (typical: 0.95–0.99), with an effective window of `1/(1−λ)` samples.
/// The update uses the effective sample weight, so `lambda = 1` is a
/// true cumulative mean rather than a frozen one.
#[derive(Debug, Clone)]
pub struct InnovationTracker {
    /// Forgetting factor in (0, 1].
    pub lambda: f64,
    /// Exponentially weighted mean of ν².
    pub mean_nu_sq: f64,
    /// Exponentially weighted mean of P⁻.
    pub mean_p_pred: f64,
    /// Exponentially weighted mean of NIS = ν² / (P⁻ + R).
    pub mean_nis: f64,
    /// Number of innovations seen (effective count saturates at 1/(1−λ)).
    pub count: usize,
    /// Effective sample weight (grows to 1/(1−λ); equals `count` for λ=1).
    weight: f64,
    /// Whether any innovation has been seen yet.
    pub initialised: bool,
}

impl InnovationTracker {
    /// Create a tracker. `lambda` must be in (0, 1].
    pub fn new(lambda: f64) -> Result<Self> {
        if !(0.0 < lambda && lambda <= 1.0) || !lambda.is_finite() {
            return Err(crate::errors::KalmanError::InvalidParameter(format!(
                "lambda must be in (0, 1], got {lambda}"
            )));
        }
        Ok(Self {
            lambda,
            mean_nu_sq: 0.0,
            mean_p_pred: 0.0,
            mean_nis: 0.0,
            count: 0,
            weight: 0.0,
            initialised: false,
        })
    }

    /// Feed one innovation: `nu` (residual), `p_pred` (a priori
    /// variance), `r` (current measurement-noise variance).
    pub fn observe(&mut self, nu: f64, p_pred: f64, r: f64) {
        let nu_sq = nu * nu;
        let nis = nu_sq / (p_pred + r);
        if !self.initialised {
            self.mean_nu_sq = nu_sq;
            self.mean_p_pred = p_pred;
            self.mean_nis = nis;
            self.weight = 1.0;
            self.initialised = true;
        } else {
            // Effective-weight update: with λ = 1 the weight grows
            // 1, 2, 3, … (cumulative mean); with λ < 1 it saturates at
            // 1/(1−λ) and the old mean keeps weight λ.
            let old_weight = self.weight;
            self.weight = self.lambda * self.weight + 1.0;
            let a = self.lambda * old_weight / self.weight;
            self.mean_nu_sq = a * self.mean_nu_sq + (1.0 - a) * nu_sq;
            self.mean_p_pred = a * self.mean_p_pred + (1.0 - a) * p_pred;
            self.mean_nis = a * self.mean_nis + (1.0 - a) * nis;
        }
        self.count += 1;
    }

    /// Covariance-matching estimate of R: `mean(ν²) − mean(P⁻)`,
    /// floored at a small positive value so the filter never divides
    /// by zero. Returns `None` before any observation.
    pub fn estimate_r(&self, floor: f64) -> Option<f64> {
        if !self.initialised {
            return None;
        }
        Some((self.mean_nu_sq - self.mean_p_pred).max(floor))
    }

    /// Mean NIS, clamped to `[1/10, 10]` so one bad window cannot
    /// dominate the tuning. If the filter is consistent the value is 1;
    /// NIS > 1 means the state wanders more than the model allows, so Q
    /// should grow; NIS < 1 means Q should shrink. Callers apply the
    /// correction in small damped steps (see the CLI's `--adaptive`
    /// mode): applying the raw factor in one jump is unstable, because
    /// a larger Q inflates `mean(P⁻)`, which drives the R̂ estimate
    /// down, which collapses the gain — the feedback diverges.
    /// Returns `None` before any observation.
    pub fn q_scale_factor(&self) -> Option<f64> {
        if !self.initialised {
            return None;
        }
        Some(self.mean_nis.clamp(0.1, 10.0))
    }
}

/// Batch covariance-matching estimate of R from a finished run.
///
/// Given the innovations and a priori variances of every step,
/// `R̂ = mean(ν²) − mean(P⁻)` (Mehra, 1970). Returns `None` for an
/// empty input. The result is floored at `floor` (default: 1e-12).
pub fn estimate_r_batch(
    innovations: &[f64],
    p_preds: &[f64],
    floor: f64,
) -> Option<f64> {
    if innovations.is_empty() || innovations.len() != p_preds.len() {
        return None;
    }
    let n = innovations.len() as f64;
    let mean_nu_sq: f64 = innovations.iter().map(|v| v * v).sum::<f64>() / n;
    let mean_p_pred: f64 = p_preds.iter().sum::<f64>() / n;
    Some((mean_nu_sq - mean_p_pred).max(floor))
}

/// Batch estimate of the mean NIS over a finished run.
pub fn mean_nis_batch(innovations: &[f64], innovation_variances: &[f64]) -> Option<f64> {
    if innovations.is_empty() || innovations.len() != innovation_variances.len() {
        return None;
    }
    let n = innovations.len() as f64;
    let total: f64 = innovations
        .iter()
        .zip(innovation_variances.iter())
        .map(|(nu, s)| nu * nu / s)
        .sum();
    Some(total / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_recovers_r_from_well_tuned_filter() {
        // Simulate a well-tuned scalar filter (true R = 0.25) and feed
        // its innovations to the tracker: R̂ must land near 0.25.
        // 4000 samples keep the Monte-Carlo error small (~±0.02).
        let mut kf = crate::scalar::ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
        let mut rng = crate::rng::GaussianRng::new(20261121);
        let mut tracker = InnovationTracker::new(1.0).unwrap();
        let mut truth = 0.0f64;
        for _ in 0..4000 {
            truth += rng.gaussian(0.0, 0.1); // process noise sqrt(Q)
            let z = truth + rng.gaussian(0.0, 0.5); // meas. noise sqrt(R)
            let step = kf.step_plain(z);
            tracker.observe(step.innovation, step.p_pred, 0.25);
        }
        let r_hat = tracker.estimate_r(1e-12).unwrap();
        assert!(
            (r_hat - 0.25).abs() < 0.05,
            "R̂ = {r_hat}, expected ≈ 0.25"
        );
        // A well-tuned filter has mean NIS ≈ 1.
        let nis = tracker.mean_nis;
        assert!((nis - 1.0).abs() < 0.1, "mean NIS = {nis}");
    }

    #[test]
    fn tracker_detects_mis_tuned_filter() {
        // True R = 1.0 but the filter assumes R = 0.01: the innovations
        // are far larger than the filter expects, so mean NIS ≫ 1 and
        // the Q scale factor must exceed 1.
        let mut kf = crate::scalar::ScalarKalman::new(0.0, 1.0, 0.01, 0.01).unwrap();
        let mut rng = crate::rng::GaussianRng::new(555);
        let mut tracker = InnovationTracker::new(1.0).unwrap();
        let mut truth = 0.0f64;
        for _ in 0..2000 {
            truth += rng.gaussian(0.0, 0.1);
            let z = truth + rng.gaussian(0.0, 1.0);
            let step = kf.step_plain(z);
            tracker.observe(step.innovation, step.p_pred, 0.01);
        }
        assert!(tracker.mean_nis > 2.0, "mean NIS = {}", tracker.mean_nis);
        assert!(tracker.q_scale_factor().unwrap() > 1.0);
    }

    #[test]
    fn batch_estimate_matches_tracker() {
        let innovations = vec![0.5, -0.3, 0.2, -0.1];
        let p_preds = vec![0.1, 0.1, 0.1, 0.1];
        // mean(ν²) = (0.25+0.09+0.04+0.01)/4 = 0.0975; R̂ = 0.0975 − 0.1 < 0
        // → floored at 1e-6.
        let r = estimate_r_batch(&innovations, &p_preds, 1e-6).unwrap();
        assert!((r - 1e-6).abs() < 1e-12);

        let innovations = vec![1.0, -1.0, 1.0, -1.0];
        let r = estimate_r_batch(&innovations, &p_preds, 1e-6).unwrap();
        assert!((r - 0.9).abs() < 1e-12);
    }

    #[test]
    fn mean_nis_batch_matches_hand_calculation() {
        // ν²/S = 1/2, 4/2 → mean 1.25.
        let nis = mean_nis_batch(&[1.0, 2.0], &[2.0, 2.0]).unwrap();
        assert!((nis - 1.25).abs() < 1e-12);
    }

    #[test]
    fn tracker_rejects_bad_lambda() {
        assert!(InnovationTracker::new(0.0).is_err());
        assert!(InnovationTracker::new(1.5).is_err());
        assert!(InnovationTracker::new(1.0).is_ok());
    }

    #[test]
    fn tracker_reports_none_before_any_observation() {
        let t = InnovationTracker::new(0.99).unwrap();
        assert_eq!(t.estimate_r(1e-6), None);
        assert_eq!(t.q_scale_factor(), None);
    }
}
