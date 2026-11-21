//! Integration tests for the metrics module: RMSE/NEES on full
//! synthetic runs and the chi-square quantile table used by every
//! consistency verdict in the CLI.

use kalman_filter_1d::metrics::{chi2_quantile, mae, nees_report, rmse};

#[test]
fn rmse_and_mae_agree_on_constant_error() {
    let est = vec![1.0, 1.0, 1.0];
    let truth = vec![0.0, 0.0, 0.0];
    assert!((rmse(&est, &truth).unwrap() - 1.0).abs() < 1e-12);
    assert!((mae(&est, &truth).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn chi2_median_is_close_to_dof_minus_two_thirds() {
    // Wilson–Hilferty median approximation: χ²_0.5(ν) ≈ ν(1 − 2/(9ν))³.
    // The exact quantile must be within 1% of it for ν ≥ 10.
    for dof in [10.0, 50.0, 100.0, 500.0] {
        let median = chi2_quantile(0.5, dof);
        let approx = dof * (1.0 - 2.0 / (9.0 * dof)).powi(3);
        let rel = ((median - approx) / approx).abs();
        assert!(rel < 0.01, "dof {dof}: median {median} vs approx {approx}");
    }
}

#[test]
fn nees_report_interval_widens_for_few_samples() {
    // With 10 samples the 95% interval is wide; with 10 000 it is
    // narrow. Both must contain 1.0 (the mean of a consistent filter).
    let rep_few = nees_report(&[1.0; 10], 1).unwrap();
    let rep_many = nees_report(&[1.0; 10_000], 1).unwrap();
    assert!(rep_few.lower_95 < rep_many.lower_95);
    assert!(rep_few.upper_95 > rep_many.upper_95);
    assert!(rep_few.consistent_95 && rep_many.consistent_95);
}
