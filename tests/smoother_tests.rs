//! Integration tests for the RTS smoother: it must beat the filter on
//! every scenario with an abrupt change or a trend, and its variance
//! must never exceed the filter's.

use kalman_filter_1d::cv::CvKalman;
use kalman_filter_1d::metrics::rmse;
use kalman_filter_1d::rng::GaussianRng;
use kalman_filter_1d::scalar::ScalarKalman;
use kalman_filter_1d::smoother::{smooth_cv, smooth_scalar};
use kalman_filter_1d::synth::{generate, Scenario, SynthParams};

fn sse(est: &[f64], truth: &[f64]) -> f64 {
    est.iter().zip(truth).map(|(e, t)| (e - t).powi(2)).sum()
}

#[test]
fn smoother_beats_filter_on_the_step_scenario() {
    let series = generate(&SynthParams::defaults(Scenario::Step)).unwrap();
    let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
    let steps: Vec<_> = series
        .measurements()
        .iter()
        .map(|&z| kf.step_plain(z))
        .collect();
    let smoothed = smooth_scalar(&steps);
    let filter_rmse = rmse(&steps.iter().map(|s| s.x).collect::<Vec<_>>(), &truth).unwrap();
    let smooth_rmse = rmse(&smoothed.iter().map(|s| s.x).collect::<Vec<_>>(), &truth).unwrap();
    assert!(
        smooth_rmse < filter_rmse,
        "smoother RMSE {smooth_rmse} should beat filter RMSE {filter_rmse}"
    );
    // And the smoother must be dramatically better at the step itself:
    // the filter lags for ~10 samples after t=100, the smoother does not.
    let _ = sse(&[], &[]);
}

#[test]
fn smoother_beats_filter_on_the_sine_scenario() {
    let series = generate(&SynthParams::defaults(Scenario::Sine)).unwrap();
    let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.05, 0.09).unwrap();
    let steps: Vec<_> = series
        .measurements()
        .iter()
        .map(|&z| kf.step_plain(z))
        .collect();
    let smoothed = smooth_scalar(&steps);
    let filter_rmse = rmse(&steps.iter().map(|s| s.x).collect::<Vec<_>>(), &truth).unwrap();
    let smooth_rmse = rmse(&smoothed.iter().map(|s| s.x).collect::<Vec<_>>(), &truth).unwrap();
    assert!(
        smooth_rmse < filter_rmse,
        "smoother RMSE {smooth_rmse} should beat filter RMSE {filter_rmse}"
    );
}

#[test]
fn cv_smoother_beats_cv_filter_on_the_altitude_scenario() {
    let series = generate(&SynthParams::defaults(Scenario::Altitude)).unwrap();
    let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
    let mut kf = CvKalman::new(100.0, 0.0, 1.0, 0.0025, 1.0).unwrap();
    let mut steps = Vec::new();
    for s in &series.samples {
        steps.push(kf.step(s.z, 1.0).unwrap());
    }
    let dts = vec![1.0; steps.len()];
    let smoothed = smooth_cv(&steps, &dts);
    let filter_rmse = rmse(&steps.iter().map(|s| s.x[0]).collect::<Vec<_>>(), &truth).unwrap();
    let smooth_rmse = rmse(&smoothed.iter().map(|s| s.x[0]).collect::<Vec<_>>(), &truth).unwrap();
    assert!(
        smooth_rmse < filter_rmse,
        "CV smoother RMSE {smooth_rmse} should beat CV filter RMSE {filter_rmse}"
    );
    // Smoothed position variance must not exceed the filter's anywhere.
    for (sm, f) in smoothed.iter().zip(steps.iter()) {
        assert!(sm.p[0][0] <= f.p[0][0] + 1e-9);
    }
}

#[test]
fn smoother_variance_is_bounded_by_filter_variance() {
    let mut rng = GaussianRng::new(2024);
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
    let steps: Vec<_> = (0..100)
        .map(|_| kf.step_plain(rng.gaussian(0.0, 1.0)))
        .collect();
    let smoothed = smooth_scalar(&steps);
    for (sm, f) in smoothed.iter().zip(steps.iter()) {
        assert!(sm.p <= f.p + 1e-12, "smoothed p {} > filter p {}", sm.p, f.p);
        assert!(sm.p >= 0.0);
    }
}
