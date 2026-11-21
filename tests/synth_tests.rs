//! Integration tests for the synthetic generators: every scenario must
//! be reproducible, carry ground truth, and have the statistical
//! properties its documentation promises.

use kalman_filter_1d::metrics::rmse;
use kalman_filter_1d::synth::{generate, Scenario, SynthParams};

#[test]
fn every_scenario_is_reproducible_and_has_truth() {
    for scenario in [
        Scenario::Step,
        Scenario::Sine,
        Scenario::Ramp,
        Scenario::Temperature,
        Scenario::Altitude,
    ] {
        let p = SynthParams::defaults(scenario);
        let a = generate(&p).unwrap();
        let b = generate(&p).unwrap();
        assert_eq!(a, b, "scenario {scenario:?} is not reproducible");
        assert!(a.has_full_truth());
        assert_eq!(a.len(), p.n);
    }
}

#[test]
fn raw_measurement_rmse_matches_the_noise_std() {
    // With N samples the raw RMSE concentrates on the noise std:
    // |RMSE − σ| < 3σ/√(2N) with overwhelming probability.
    // N = 5000, σ = 0.4 → tolerance ≈ 0.017.
    let mut p = SynthParams::defaults(Scenario::Ramp);
    p.n = 5000;
    p.noise_std = 0.4;
    let series = generate(&p).unwrap();
    let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
    let raw_rmse = rmse(&series.measurements(), &truth).unwrap();
    assert!(
        (raw_rmse - 0.4).abs() < 0.02,
        "raw RMSE {raw_rmse} should be ≈ noise std 0.4"
    );
}

#[test]
fn step_scenario_raw_rmse_is_the_noise_std_outside_the_jump() {
    // The raw RMSE over the whole step series is ≈ σ (the jump itself
    // does not affect the raw measurements, only the truth changes).
    let series = generate(&SynthParams::defaults(Scenario::Step)).unwrap();
    let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
    let raw_rmse = rmse(&series.measurements(), &truth).unwrap();
    assert!(
        (raw_rmse - 0.5).abs() < 0.1,
        "raw RMSE {raw_rmse} should be ≈ 0.5"
    );
}
