//! Integration tests for the constant-velocity filter: tracking quality
//! and NEES consistency (position-only, 1 dof, and full 2-dof when
//! velocity truth is known from the synthetic altitude generator).

use kalman_filter_1d::cv::CvKalman;
use kalman_filter_1d::metrics::{nees_cv, nees_report, nees_scalar, rmse};
use kalman_filter_1d::rng::GaussianRng;

#[test]
fn cv_filter_tracks_a_ramp_better_than_the_scalar_filter() {
    // True signal x = 0.5·t, 100 samples, measurement noise σ = 0.5.
    // The CV filter carries the trend in its velocity state; the
    // scalar random-walk filter lags by a constant amount.
    let mut rng = GaussianRng::new(31415);
    let mut zs = Vec::with_capacity(100);
    let mut truth = Vec::with_capacity(100);
    for k in 0..100 {
        let t = 0.5 * k as f64;
        truth.push(t);
        zs.push(t + rng.gaussian(0.0, 0.5));
    }
    let mut cv = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.25).unwrap();
    let mut cv_est = Vec::with_capacity(100);
    for &z in &zs {
        cv_est.push(cv.step(z, 1.0).unwrap().x[0]);
    }
    let mut sc = kalman_filter_1d::scalar::ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
    let mut sc_est = Vec::with_capacity(100);
    for &z in &zs {
        sc_est.push(sc.step_plain(z).x);
    }
    let cv_rmse = rmse(&cv_est, &truth).unwrap();
    let sc_rmse = rmse(&sc_est, &truth).unwrap();
    assert!(
        cv_rmse < sc_rmse,
        "CV RMSE {cv_rmse} should beat scalar RMSE {sc_rmse} on a ramp"
    );
    assert!(cv_rmse < 0.35, "CV RMSE {cv_rmse} too large");
}

#[test]
fn cv_filter_is_position_nees_consistent() {
    // Constant-velocity truth with matched noise: position NEES
    // (1 dof, marginal variance P₀₀) must be consistent over 1000 steps.
    let mut rng = GaussianRng::new(2718);
    let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.25).unwrap();
    let mut nees = Vec::with_capacity(1000);
    let mut x = 0.0f64;
    let mut v = 0.5f64;
    for _ in 0..1000 {
        // True continuous white-noise acceleration motion, matched to
        // the filter's Q = q·[[dt⁴/4, dt³/2],[dt³/2, dt²]] with dt = 1:
        // x += v + a/2, v += a, a ~ N(0, q) with q = 0.01.
        let a = rng.gaussian(0.0, 0.1);
        x += v + 0.5 * a;
        v += a;
        let z = x + rng.gaussian(0.0, 0.5);
        let step = kf.step(z, 1.0).unwrap();
        nees.push(nees_scalar(step.x[0], x, step.p[0][0]).unwrap());
    }
    let rep = nees_report(&nees, 1).unwrap();
    assert!(
        rep.consistent_95,
        "mean NEES {} outside [{:.4}, {:.4}]",
        rep.mean, rep.lower_95, rep.upper_95
    );
}

#[test]
fn cv_filter_full_state_nees_is_consistent() {
    // Same run, but the full 2-dof NEES against position+velocity truth.
    let mut rng = GaussianRng::new(1618);
    let mut kf = CvKalman::new(0.0, 0.0, 1.0, 0.01, 0.25).unwrap();
    let mut nees = Vec::with_capacity(1000);
    let mut x = 0.0f64;
    let mut v = 0.5f64;
    for _ in 0..1000 {
        // Same matched continuous-acceleration motion as above.
        let a = rng.gaussian(0.0, 0.1);
        x += v + 0.5 * a;
        v += a;
        let z = x + rng.gaussian(0.0, 0.5);
        let step = kf.step(z, 1.0).unwrap();
        nees.push(nees_cv(step.x, [x, v], step.p).unwrap());
    }
    let rep = nees_report(&nees, 2).unwrap();
    assert!(
        rep.consistent_95,
        "mean NEES {} outside [{:.4}, {:.4}]",
        rep.mean, rep.lower_95, rep.upper_95
    );
}
