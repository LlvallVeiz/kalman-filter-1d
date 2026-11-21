//! Integration tests for the α–β tracker: it must track, and the
//! Kalata steady-state gains must reproduce the Kalman filter's
//! steady-state behaviour on a constant signal.

use kalman_filter_1d::alpha_beta::{alpha_beta_from_lambda, AlphaBeta};
use kalman_filter_1d::metrics::rmse;
use kalman_filter_1d::scalar::ScalarKalman;

#[test]
fn alpha_beta_tracks_a_ramp() {
    let mut f = AlphaBeta::new(0.0, 0.0, 0.5, 0.1).unwrap();
    let mut est = Vec::new();
    let mut truth = Vec::new();
    for k in 0..100 {
        let t = 0.3 * k as f64;
        truth.push(t);
        est.push(f.step(t, 1.0).unwrap().x);
    }
    let e = rmse(&est, &truth).unwrap();
    assert!(e < 0.5, "alpha-beta ramp RMSE {e}");
}

#[test]
fn kalata_gains_span_the_useful_range() {
    // λ = 0.05 (very smooth) → small gains; λ = 1.9 (very reactive)
    // → gains near the α=1, β=2 saturation.
    let (a_lo, b_lo) = alpha_beta_from_lambda(0.05);
    let (a_hi, b_hi) = alpha_beta_from_lambda(1.9);
    assert!(a_lo < 0.5 && b_lo < 0.2, "λ=0.05: α={a_lo}, β={b_lo}");
    assert!(a_hi > 0.95 && b_hi > 1.8, "λ=1.9: α={a_hi}, β={b_hi}");
}

#[test]
fn alpha_beta_with_kalman_steady_state_gains_matches_kalman() {
    // On a long constant signal the scalar Kalman filter converges to
    // its steady-state gain K∞; an α–β tracker fed the same constant
    // gain must converge to the same estimate (both are first-order
    // low-pass filters with the same pole once settled).
    // Q = 0.01, R = 0.1 → K∞ = 0.2701562119.
    let kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
    let k_ss = kf.steady_state_gain();
    let mut ab = AlphaBeta::new(0.0, 0.0, k_ss, 0.0 + 1e-9).unwrap();
    // β must be > 0; use a tiny β so the velocity state stays ~0 and
    // the tracker behaves like the scalar filter.
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
    for _ in 0..500 {
        let _ = kf.step_plain(5.0);
        let _ = ab.step(5.0, 1.0).unwrap();
    }
    assert!(
        (kf.x - ab.x).abs() < 0.05,
        "Kalman x={} vs alpha-beta x={} (K∞={k_ss})",
        kf.x, ab.x
    );
}
