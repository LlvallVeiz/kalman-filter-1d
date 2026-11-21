//! Integration tests for adaptive noise estimation: a filter that
//! starts mis-tuned must recover a usable R estimate from its own
//! innovations, and the batch estimators must agree with the tracker.

use kalman_filter_1d::noise::{estimate_r_batch, mean_nis_batch, InnovationTracker};
use kalman_filter_1d::rng::GaussianRng;
use kalman_filter_1d::scalar::ScalarKalman;

#[test]
fn tracker_recovers_r_after_a_mistuned_start() {
    // Start with a wrong R (0.01 instead of the true 1.0) and let the
    // tracker watch 3000 innovations: R̂ must move toward 1.0.
    // The filter itself stays mis-tuned (we only watch), so the
    // innovations stay large and the estimate is biased upward a
    // little; the test only requires the right order of magnitude.
    let mut rng = GaussianRng::new(60606);
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.01).unwrap();
    let mut tracker = InnovationTracker::new(1.0).unwrap();
    let mut truth = 0.0f64;
    for _ in 0..3000 {
        truth += rng.gaussian(0.0, 0.1);
        let z = truth + rng.gaussian(0.0, 1.0);
        let step = kf.step_plain(z);
        tracker.observe(step.innovation, step.p_pred, 0.01);
    }
    let r_hat = tracker.estimate_r(1e-12).unwrap();
    assert!(
        (0.3..3.0).contains(&r_hat),
        "R̂ = {r_hat}, expected the order of the true R = 1.0"
    );
}

#[test]
fn batch_estimate_agrees_with_tracker_on_the_same_run() {
    let mut rng = GaussianRng::new(13579);
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
    let mut tracker = InnovationTracker::new(1.0).unwrap();
    let mut innovations = Vec::new();
    let mut p_preds = Vec::new();
    let mut truth = 0.0f64;
    for _ in 0..2000 {
        truth += rng.gaussian(0.0, 0.1);
        let z = truth + rng.gaussian(0.0, 0.5);
        let step = kf.step_plain(z);
        tracker.observe(step.innovation, step.p_pred, 0.25);
        innovations.push(step.innovation);
        p_preds.push(step.p_pred);
    }
    let from_tracker = tracker.estimate_r(1e-12).unwrap();
    let from_batch = estimate_r_batch(&innovations, &p_preds, 1e-12).unwrap();
    assert!(
        (from_tracker - from_batch).abs() < 1e-9,
        "tracker R̂ {from_tracker} vs batch R̂ {from_batch}"
    );
    let nis_tracker = tracker.mean_nis;
    let nis_batch = mean_nis_batch(
        &innovations,
        &innovations
            .iter()
            .zip(p_preds.iter())
            .map(|(_, p)| p + 0.25)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(
        (nis_tracker - nis_batch).abs() < 1e-9,
        "tracker NIS {nis_tracker} vs batch NIS {nis_batch}"
    );
}

#[test]
fn forgetting_tracker_follows_a_change_in_noise() {
    // R jumps from 0.25 to 4.0 halfway through. A tracker with
    // λ = 0.99 must end up estimating a much larger R than a tracker
    // with λ = 1.0 (which averages the whole history).
    let mut rng = GaussianRng::new(24680);
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
    let mut fast = InnovationTracker::new(0.99).unwrap();
    let mut slow = InnovationTracker::new(1.0).unwrap();
    let mut truth = 0.0f64;
    for k in 0..4000 {
        truth += rng.gaussian(0.0, 0.1);
        let noise_std = if k < 2000 { 0.5 } else { 2.0 };
        let z = truth + rng.gaussian(0.0, noise_std);
        let step = kf.step_plain(z);
        fast.observe(step.innovation, step.p_pred, 0.25);
        slow.observe(step.innovation, step.p_pred, 0.25);
    }
    let r_fast = fast.estimate_r(1e-12).unwrap();
    let r_slow = slow.estimate_r(1e-12).unwrap();
    assert!(
        r_fast > r_slow,
        "forgetting tracker R̂ {r_fast} should exceed full-history R̂ {r_slow}"
    );
    assert!(r_fast > 1.0, "forgetting tracker R̂ {r_fast} should be near 4.0");
}
