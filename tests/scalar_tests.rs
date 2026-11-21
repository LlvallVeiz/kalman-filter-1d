//! Integration tests for the scalar Kalman filter: end-to-end behaviour
//! through the public API, including the NEES consistency property that
//! only shows up over long runs.

use kalman_filter_1d::metrics::{nees_report, nees_scalar};
use kalman_filter_1d::rng::GaussianRng;
use kalman_filter_1d::scalar::{run_scalar_series, ScalarKalman};

#[test]
fn well_tuned_filter_is_nees_consistent_over_a_long_run() {
    // 2000 samples, true Q = 0.01, true R = 0.25, filter matched.
    // Mean NEES must lie inside the 95% interval for 2000 dof
    // (≈ [0.94, 1.06]); a mis-tuned filter would fall far outside.
    let mut rng = GaussianRng::new(777);
    let mut truth = 0.0f64;
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
    let mut nees = Vec::with_capacity(2000);
    for _ in 0..2000 {
        truth += rng.gaussian(0.0, 0.1);
        let z = truth + rng.gaussian(0.0, 0.5);
        let step = kf.step_plain(z);
        nees.push(nees_scalar(step.x, truth, step.p).unwrap());
    }
    let rep = nees_report(&nees, 1).unwrap();
    assert!(
        rep.consistent_95,
        "mean NEES {} outside [{:.4}, {:.4}]",
        rep.mean, rep.lower_95, rep.upper_95
    );
}

#[test]
fn mistuned_q_makes_the_filter_overconfident() {
    // True Q = 0.25 (fast wandering state) but the filter assumes
    // Q = 0.0001: its covariance is far too small, so the mean NEES
    // must blow up well above 1.
    let mut rng = GaussianRng::new(888);
    let mut truth = 0.0f64;
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.0001, 0.25).unwrap();
    let mut nees = Vec::with_capacity(2000);
    for _ in 0..2000 {
        truth += rng.gaussian(0.0, 0.5);
        let z = truth + rng.gaussian(0.0, 0.5);
        let step = kf.step_plain(z);
        nees.push(nees_scalar(step.x, truth, step.p).unwrap());
    }
    let rep = nees_report(&nees, 1).unwrap();
    assert!(!rep.consistent_95, "mean NEES {} should be inconsistent", rep.mean);
    assert!(rep.mean > 2.0, "mean NEES {} should be far above 1", rep.mean);
}

#[test]
fn run_scalar_series_matches_step_by_step() {
    let kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
    let steps = run_scalar_series(kf, &[1.0, 1.1, 0.9]);
    assert_eq!(steps.len(), 3);
    assert!((steps[0].x - 0.909_909_909_9).abs() < 1e-9);
    assert!((steps[1].x - 1.005_423_576_9).abs() < 1e-9);
    assert!((steps[2].x - 0.965_788_369_6).abs() < 1e-9);
}
