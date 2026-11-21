//! Rauch–Tung–Striebel (RTS) smoother — fixed-interval smoothing.
//!
//! A filter estimates `x_k` using only measurements up to time `k`.
//! A smoother re-processes the whole batch *backwards* and estimates
//! `x_k` using **all** measurements, past and future, so it is strictly
//! more accurate than the filter at every interior point (Rauch, Tung &
//! Striebel, 1965). The price: it needs the complete forward pass first
//! and cannot run online.
//!
//! Backward recursion (scalar form; the CV form is the matrix analogue):
//!
//! ```text
//!   G_k   = P_k · Fᵀ / P⁻_{k+1}            (smoother gain)
//!   x̂ˢ_k = x̂_k + G_k (x̂ˢ_{k+1} − x̂⁻_{k+1})
//!   Pˢ_k  = P_k + G_k² (Pˢ_{k+1} − P⁻_{k+1})
//! ```
//!
//! with `x̂ˢ_N = x̂_N`, `Pˢ_N = P_N` at the last sample. The smoother
//! variance is never larger than the filter variance, and it is
//! strictly smaller whenever the future carries information
//! (`Pˢ_{k+1} < P⁻_{k+1}`).

use crate::cv::CvStep;
use crate::scalar::ScalarStep;

/// One smoothed scalar sample: the RTS estimate and its variance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothedScalar {
    /// Smoothed state estimate.
    pub x: f64,
    /// Smoothed error variance (≤ filter variance).
    pub p: f64,
    /// Smoother gain used at this step.
    pub gain: f64,
}

/// RTS smoothing over a completed scalar forward pass.
///
/// `steps` is the full sequence of [`ScalarStep`] records (one per
/// measurement, in time order). Returns one [`SmoothedScalar`] per step.
/// The last smoothed value equals the last filtered value.
pub fn smooth_scalar(steps: &[ScalarStep]) -> Vec<SmoothedScalar> {
    let n = steps.len();
    let mut out = vec![
        SmoothedScalar {
            x: 0.0,
            p: 0.0,
            gain: 0.0,
        };
        n
    ];
    if n == 0 {
        return out;
    }
    // Anchor: the last smoothed estimate is the last filtered estimate.
    out[n - 1] = SmoothedScalar {
        x: steps[n - 1].x,
        p: steps[n - 1].p,
        gain: 0.0,
    };
    // Backward pass. F = 1 for the random walk, so G = P_k / P⁻_{k+1}.
    for k in (0..n - 1).rev() {
        let p_pred_next = steps[k + 1].p_pred;
        let gain = if p_pred_next > 0.0 {
            steps[k].p / p_pred_next
        } else {
            0.0
        };
        let x = steps[k].x + gain * (out[k + 1].x - steps[k + 1].x_pred);
        let p = steps[k].p + gain * gain * (out[k + 1].p - p_pred_next);
        out[k] = SmoothedScalar { x, p, gain };
    }
    out
}

/// One smoothed CV sample: position, velocity and covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothedCv {
    /// Smoothed state `[position, velocity]`.
    pub x: [f64; 2],
    /// Smoothed 2×2 error covariance.
    pub p: [[f64; 2]; 2],
    /// Smoother gain matrix (2×2).
    pub gain: [[f64; 2]; 2],
}

/// RTS smoothing over a completed CV forward pass.
///
/// `dts[i]` is the step length that led *into* sample `i`
/// (`dts[0]` is unused — the first sample has no predecessor — but must
/// still be present so the slice aligns with `steps`).
/// `f(dt) = [[1, dt],[0, 1]]`.
pub fn smooth_cv(steps: &[CvStep], dts: &[f64]) -> Vec<SmoothedCv> {
    let n = steps.len();
    let mut out = vec![
        SmoothedCv {
            x: [0.0, 0.0],
            p: [[0.0, 0.0], [0.0, 0.0]],
            gain: [[0.0, 0.0], [0.0, 0.0]],
        };
        n
    ];
    assert_eq!(
        dts.len(),
        n,
        "dts must have one entry per step (dts[0] unused)"
    );
    if n == 0 {
        return out;
    }
    out[n - 1] = SmoothedCv {
        x: steps[n - 1].x,
        p: steps[n - 1].p,
        gain: [[0.0, 0.0], [0.0, 0.0]],
    };
    for k in (0..n - 1).rev() {
        let dt = dts[k + 1];
        // G = P_k · Fᵀ · (P⁻_{k+1})⁻¹,  F = [[1, dt],[0, 1]].
        // P⁻_{k+1} is 2×2 symmetric; invert it in closed form.
        let pp = steps[k + 1].p_pred;
        let det = pp[0][0] * pp[1][1] - pp[0][1] * pp[1][0];
        let gain = if det.abs() < 1e-300 {
            [[0.0, 0.0], [0.0, 0.0]]
        } else {
            // P⁻¹ = (1/det)·[[P₁₁, −P₀₁],[−P₁₀, P₀₀]]
            let inv = [
                [pp[1][1] / det, -pp[0][1] / det],
                [-pp[1][0] / det, pp[0][0] / det],
            ];
            // P_k · Fᵀ : Fᵀ = [[1, 0],[dt, 1]]
            let pk = steps[k].p;
            let pf_t = [
                [pk[0][0] + pk[0][1] * dt, pk[0][1]],
                [pk[1][0] + pk[1][1] * dt, pk[1][1]],
            ];
            // (P_k·Fᵀ)·P⁻¹
            [
                [
                    pf_t[0][0] * inv[0][0] + pf_t[0][1] * inv[1][0],
                    pf_t[0][0] * inv[0][1] + pf_t[0][1] * inv[1][1],
                ],
                [
                    pf_t[1][0] * inv[0][0] + pf_t[1][1] * inv[1][0],
                    pf_t[1][0] * inv[0][1] + pf_t[1][1] * inv[1][1],
                ],
            ]
        };
        // x̂ˢ_k = x̂_k + G·(x̂ˢ_{k+1} − x̂⁻_{k+1})
        let dx = [
            out[k + 1].x[0] - steps[k + 1].x_pred[0],
            out[k + 1].x[1] - steps[k + 1].x_pred[1],
        ];
        let x = [
            steps[k].x[0] + gain[0][0] * dx[0] + gain[0][1] * dx[1],
            steps[k].x[1] + gain[1][0] * dx[0] + gain[1][1] * dx[1],
        ];
        // Pˢ_k = P_k + G·(Pˢ_{k+1} − P⁻_{k+1})·Gᵀ
        let dp = [
            [
                out[k + 1].p[0][0] - pp[0][0],
                out[k + 1].p[0][1] - pp[0][1],
            ],
            [
                out[k + 1].p[1][0] - pp[1][0],
                out[k + 1].p[1][1] - pp[1][1],
            ],
        ];
        // M = G·DP, then P = P_k + M·Gᵀ.
        let m = [
            [
                gain[0][0] * dp[0][0] + gain[0][1] * dp[1][0],
                gain[0][0] * dp[0][1] + gain[0][1] * dp[1][1],
            ],
            [
                gain[1][0] * dp[0][0] + gain[1][1] * dp[1][0],
                gain[1][0] * dp[0][1] + gain[1][1] * dp[1][1],
            ],
        ];
        let pk = steps[k].p;
        let p = [
            [
                pk[0][0] + m[0][0] * gain[0][0] + m[0][1] * gain[0][1],
                pk[0][1] + m[0][0] * gain[1][0] + m[0][1] * gain[1][1],
            ],
            [
                pk[1][0] + m[1][0] * gain[0][0] + m[1][1] * gain[0][1],
                pk[1][1] + m[1][0] * gain[1][0] + m[1][1] * gain[1][1],
            ],
        ];
        out[k] = SmoothedCv { x, p, gain };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::CvKalman;
    use crate::scalar::ScalarKalman;

    #[test]
    fn scalar_smoother_never_worsens_variance() {
        let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
        let zs = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03];
        let steps: Vec<_> = zs.iter().map(|&z| kf.step_plain(z)).collect();
        let smoothed = smooth_scalar(&steps);
        assert_eq!(smoothed.len(), steps.len());
        for (s, f) in smoothed.iter().zip(steps.iter()) {
            assert!(
                s.p <= f.p + 1e-12,
                "smoothed variance {} > filter variance {}",
                s.p,
                f.p
            );
            assert!(s.p >= 0.0);
        }
        // Anchor: last smoothed value equals last filtered value.
        let last = smoothed.last().unwrap();
        assert!((last.x - steps.last().unwrap().x).abs() < 1e-15);
        assert!((last.p - steps.last().unwrap().p).abs() < 1e-15);
    }

    #[test]
    fn scalar_smoother_beats_filter_on_noisy_step() {
        // A step at sample 25: the filter lags, the smoother (which sees
        // the future) must have a smaller total squared error.
        let mut rng = crate::rng::GaussianRng::new(777);
        let mut truth = Vec::new();
        let mut zs = Vec::new();
        for k in 0..60 {
            let t = if k < 25 { 0.0 } else { 5.0 };
            truth.push(t);
            zs.push(t + rng.gaussian(0.0, 0.5));
        }
        let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
        let steps: Vec<_> = zs.iter().map(|&z| kf.step_plain(z)).collect();
        let smoothed = smooth_scalar(&steps);
        let err_f: f64 = steps
            .iter()
            .zip(&truth)
            .map(|(s, t)| (s.x - t).powi(2))
            .sum();
        let err_s: f64 = smoothed
            .iter()
            .zip(&truth)
            .map(|(s, t)| (s.x - t).powi(2))
            .sum();
        assert!(
            err_s < err_f,
            "smoother SSE {err_s} should beat filter SSE {err_f}"
        );
    }

    #[test]
    fn scalar_smoother_empty_input_is_empty() {
        assert!(smooth_scalar(&[]).is_empty());
    }

    #[test]
    fn cv_smoother_never_worsens_position_variance() {
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.1).unwrap();
        let dts = vec![1.0; 12];
        let mut steps = Vec::new();
        for k in 0..12 {
            let z = 0.5 * k as f64 + 0.1 * (k as f64).sin();
            steps.push(kf.step(z, 1.0).unwrap());
        }
        let smoothed = smooth_cv(&steps, &dts);
        assert_eq!(smoothed.len(), 12);
        for (s, f) in smoothed.iter().zip(steps.iter()) {
            assert!(
                s.p[0][0] <= f.p[0][0] + 1e-9,
                "smoothed P00 {} > filter P00 {}",
                s.p[0][0],
                f.p[0][0]
            );
        }
        let last = smoothed.last().unwrap();
        assert!((last.x[0] - steps.last().unwrap().x[0]).abs() < 1e-12);
    }

    #[test]
    fn cv_smoother_beats_filter_on_ramp() {
        let mut rng = crate::rng::GaussianRng::new(4242);
        let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.25).unwrap();
        let mut steps = Vec::new();
        let mut truth = Vec::new();
        for k in 0..50 {
            let t = 0.3 * k as f64;
            truth.push(t);
            steps.push(kf.step(t + rng.gaussian(0.0, 0.5), 1.0).unwrap());
        }
        let dts = vec![1.0; 50];
        let smoothed = smooth_cv(&steps, &dts);
        let err_f: f64 = steps
            .iter()
            .zip(&truth)
            .map(|(s, t)| (s.x[0] - t).powi(2))
            .sum();
        let err_s: f64 = smoothed
            .iter()
            .zip(&truth)
            .map(|(s, t)| (s.x[0] - t).powi(2))
            .sum();
        assert!(
            err_s < err_f,
            "smoother SSE {err_s} should beat filter SSE {err_f}"
        );
    }
}
