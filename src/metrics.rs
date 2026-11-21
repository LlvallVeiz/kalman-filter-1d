//! Validation metrics: RMSE and NEES consistency testing.
//!
//! Two questions every filter must answer:
//!
//! 1. **Is it accurate?** — root-mean-square error (RMSE) of the
//!    estimates against ground truth, compared with the RMSE of the raw
//!    measurements. A filter that does not beat the raw data is useless.
//! 2. **Is it consistent?** — the normalised estimation error squared
//!    (NEES), `ε_k = (x_k − x_true,k)² / P_k` for the scalar filter
//!    (and the quadratic form `(x−x_true)ᵀP⁻¹(x−x_true)` for the CV
//!    filter). If the filter's covariance is honest, each NEES sample is
//!    chi-square with `n` degrees of freedom (Bar-Shalom, Li &
//!    Kirubarajan, 2001, §5.4), so `N·ε̄` (the sample count times the
//!    mean NEES) is chi-square with `N·n` degrees of freedom, and the
//!    mean NEES must lie inside the 95% interval
//!    `[χ²_{0.025,N·n}/N, χ²_{0.975,N·n}/N]` — an interval centred on
//!    `n`, not on 1. Mean NEES ≫ n means the filter is over-confident
//!    (covariance too small, e.g. Q under-estimated); mean NEES ≪ n
//!    means it is pessimistic (covariance too large).
//!
//! Chi-square quantiles are computed exactly (to ~1e-12) by bisection on
//! the regularised lower incomplete gamma function, implemented from
//! Numerical Recipes' series / continued-fraction expansions — no
//! statistics crate, no lookup tables.

use crate::errors::{KalmanError::InsufficientData, Result};

/// Root-mean-square error between estimates and ground truth.
/// Both slices must have the same non-zero length.
pub fn rmse(estimates: &[f64], truth: &[f64]) -> Result<f64> {
    if estimates.is_empty() {
        return Err(InsufficientData("rmse needs at least one sample".into()));
    }
    if estimates.len() != truth.len() {
        return Err(InsufficientData(format!(
            "rmse: estimates ({}) and truth ({}) must have equal length",
            estimates.len(),
            truth.len()
        )));
    }
    let sse: f64 = estimates
        .iter()
        .zip(truth.iter())
        .map(|(e, t)| (e - t).powi(2))
        .sum();
    Ok((sse / estimates.len() as f64).sqrt())
}

/// Mean absolute error (a robust companion to RMSE).
pub fn mae(estimates: &[f64], truth: &[f64]) -> Result<f64> {
    if estimates.is_empty() || estimates.len() != truth.len() {
        return Err(InsufficientData(
            "mae needs two non-empty slices of equal length".into(),
        ));
    }
    let sum: f64 = estimates
        .iter()
        .zip(truth.iter())
        .map(|(e, t)| (e - t).abs())
        .sum();
    Ok(sum / estimates.len() as f64)
}

/// NEES of one scalar estimate: `(x − truth)² / p`.
/// Returns `None` when `p <= 0` (degenerate covariance).
pub fn nees_scalar(x: f64, truth: f64, p: f64) -> Option<f64> {
    if p > 0.0 {
        Some((x - truth).powi(2) / p)
    } else {
        None
    }
}

/// NEES of one 2-state (CV) estimate: `(x−x_true)ᵀ P⁻¹ (x−x_true)`.
/// Returns `None` when the covariance is singular.
pub fn nees_cv(x: [f64; 2], truth: [f64; 2], p: [[f64; 2]; 2]) -> Option<f64> {
    let det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
    if det.abs() < 1e-300 {
        return None;
    }
    let dx = [x[0] - truth[0], x[1] - truth[1]];
    // eᵀ·P⁻¹·e with P⁻¹ = (1/det)·[[P₁₁, −P₀₁],[−P₁₀, P₀₀]]
    let v0 = (p[1][1] * dx[0] - p[0][1] * dx[1]) / det;
    let v1 = (-p[1][0] * dx[0] + p[0][0] * dx[1]) / det;
    Some(dx[0] * v0 + dx[1] * v1)
}

/// Summary of a NEES consistency test over many samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeesReport {
    /// Mean NEES over the samples (should be ≈ `dof_per_sample` for a
    /// consistent filter: 1 for the scalar filter, 2 for the CV filter).
    pub mean: f64,
    /// Number of samples.
    pub n: usize,
    /// Degrees of freedom per sample (1 = scalar, 2 = CV).
    pub dof_per_sample: usize,
    /// Lower end of the 95% acceptance interval for the mean.
    pub lower_95: f64,
    /// Upper end of the 95% acceptance interval for the mean.
    pub upper_95: f64,
    /// `true` when the mean lies inside the 95% interval.
    pub consistent_95: bool,
}

/// Test the consistency of a batch of NEES values.
///
/// `dof_per_sample` is 1 for the scalar filter and 2 for the CV filter.
/// The statistic is `N · mean(NEES)`, chi-square with `N · dof`
/// degrees of freedom, so the acceptance interval for the *mean* is
/// the chi-square interval divided by the sample count `N` (not by the
/// total degrees of freedom) — the interval is centred on
/// `dof_per_sample`.
pub fn nees_report(nees_values: &[f64], dof_per_sample: usize) -> Result<NeesReport> {
    if nees_values.is_empty() {
        return Err(InsufficientData(
            "nees_report needs at least one NEES value".into(),
        ));
    }
    if dof_per_sample == 0 {
        return Err(InsufficientData(
            "dof_per_sample must be >= 1".into(),
        ));
    }
    let n = nees_values.len();
    let mean = nees_values.iter().sum::<f64>() / n as f64;
    let total_dof = (n * dof_per_sample) as f64;
    let lower_95 = chi2_quantile(0.025, total_dof) / n as f64;
    let upper_95 = chi2_quantile(0.975, total_dof) / n as f64;
    Ok(NeesReport {
        mean,
        n,
        dof_per_sample,
        lower_95,
        upper_95,
        consistent_95: mean >= lower_95 && mean <= upper_95,
    })
}

/// Quantile of the chi-square distribution with `dof` degrees of
/// freedom: the value `x` with `P(χ²_dof ≤ x) = p`.
///
/// Computed by bisection on the regularised lower incomplete gamma
/// `P(dof/2, x/2)`; 200 iterations give far more precision than any
/// validation use needs. Panics only on out-of-range arguments
/// (programmer error, not user input).
pub fn chi2_quantile(p: f64, dof: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&p),
        "chi2_quantile: p must be in (0, 1), got {p}"
    );
    assert!(dof > 0.0, "chi2_quantile: dof must be > 0, got {dof}");
    // Bracket: the quantile is below dof + 40·sqrt(2·dof) + 100 for any
    // practical p, and above 0.
    let mut lo = 0.0f64;
    let mut hi = dof + 40.0 * (2.0 * dof).sqrt() + 100.0;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if gamma_p(dof / 2.0, mid / 2.0) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Regularised lower incomplete gamma `P(a, x) = γ(a, x)/Γ(a)`.
/// Numerical Recipes `gammp`: series expansion for `x < a + 1`,
/// continued fraction (`gcf`) otherwise.
fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series representation.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction for Q(a, x); P = 1 − Q.
        1.0 - gamma_q_cf(a, x)
    }
}

/// Continued-fraction evaluation of `Q(a, x)` (Numerical Recipes `gcf`).
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const EPS: f64 = 3e-14;
    const FPMIN: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Natural log of the gamma function (Lanczos approximation,
/// Numerical Recipes `gammln`, g = 7, 9 coefficients).
// The coefficients are the canonical Lanczos values (as published);
// their extra decimal digits are intentional, hence the allows.
#[allow(clippy::excessive_precision)]
#[allow(clippy::inconsistent_digit_grouping)]
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_3116e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        let t = x + 7.5;
        for (i, &c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmse_of_perfect_estimates_is_zero() {
        assert_eq!(rmse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap(), 0.0);
    }

    #[test]
    fn rmse_matches_hand_calculation() {
        // Errors 3, 4 → RMSE = 5/√2 ≈ 3.5355.
        let r = rmse(&[3.0, 4.0], &[0.0, 0.0]).unwrap();
        assert!((r - 3.535_533_905_9).abs() < 1e-9);
    }

    #[test]
    fn rmse_rejects_mismatched_lengths() {
        assert!(rmse(&[1.0], &[1.0, 2.0]).is_err());
        assert!(rmse(&[], &[]).is_err());
    }

    #[test]
    fn nees_scalar_is_error_squared_over_variance() {
        assert_eq!(nees_scalar(3.0, 1.0, 2.0), Some(2.0));
        assert_eq!(nees_scalar(1.0, 1.0, 2.0), Some(0.0));
        assert_eq!(nees_scalar(1.0, 1.0, 0.0), None);
    }

    #[test]
    fn nees_cv_matches_scalar_when_velocity_error_is_zero() {
        // P = diag(4, 1), error (2, 0) → NEES = 4/4 = 1.
        let v = nees_cv([2.0, 0.0], [0.0, 0.0], [[4.0, 0.0], [0.0, 1.0]]);
        assert!((v.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chi2_quantiles_match_reference_values() {
        // Reference values from standard chi-square tables:
        // χ²_0.975(1) = 5.02389, χ²_0.025(1) = 0.000982,
        // χ²_0.975(2) = 7.37776, χ²_0.025(2) = 0.05064,
        // χ²_0.975(100) = 129.561, χ²_0.025(100) = 74.222
        // (cross-checked against the Poisson/gamma identity and the
        // Wilson–Hilferty approximation, which both give 74.22).
        assert!((chi2_quantile(0.975, 1.0) - 5.023_89).abs() < 1e-3);
        assert!((chi2_quantile(0.025, 1.0) - 0.000_982).abs() < 1e-4);
        assert!((chi2_quantile(0.975, 2.0) - 7.377_76).abs() < 1e-3);
        assert!((chi2_quantile(0.025, 2.0) - 0.050_64).abs() < 1e-3);
        assert!((chi2_quantile(0.975, 100.0) - 129.561).abs() < 0.05);
        assert!((chi2_quantile(0.025, 100.0) - 74.222).abs() < 0.05);
    }

    #[test]
    fn nees_report_flags_consistent_and_inconsistent() {
        // 200 samples of exactly 1.0: mean NEES 1.0 is inside the
        // 95% interval for 200 dof ([0.814, 1.205] by Wilson–Hilferty).
        let ones = vec![1.0; 200];
        let rep = nees_report(&ones, 1).unwrap();
        assert!((rep.mean - 1.0).abs() < 1e-12);
        assert!(rep.consistent_95, "mean 1.0 should be consistent");
        assert!(rep.lower_95 > 0.7 && rep.lower_95 < 0.9);
        assert!(rep.upper_95 > 1.1 && rep.upper_95 < 1.3);

        // Mean NEES 4.0 is far outside.
        let fours = vec![4.0; 200];
        let rep = nees_report(&fours, 1).unwrap();
        assert!(!rep.consistent_95);
    }

    #[test]
    fn nees_report_two_dof_interval_is_centred_on_two() {
        // A 2-dof NEES sample has expectation 2, so the 95% interval
        // for the mean of 200 such samples is centred on 2, not 1:
        // [χ²_0.025(400)/200, χ²_0.975(400)/200] ≈ [1.73, 2.29].
        let twos = vec![2.0; 200];
        let rep = nees_report(&twos, 2).unwrap();
        assert!((rep.mean - 2.0).abs() < 1e-12);
        assert!(rep.consistent_95, "mean 2.0 should be consistent for dof=2");
        assert!(rep.lower_95 > 1.6 && rep.lower_95 < 1.9, "lower {}", rep.lower_95);
        assert!(rep.upper_95 > 2.1 && rep.upper_95 < 2.4, "upper {}", rep.upper_95);
        // ...while a mean of 1.0 is inconsistent for a 2-dof filter.
        let ones = vec![1.0; 200];
        let rep = nees_report(&ones, 2).unwrap();
        assert!(!rep.consistent_95);
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        // ln Γ(1) = 0, ln Γ(0.5) = ln √π, ln Γ(5) = ln 24.
        assert!(ln_gamma(1.0).abs() < 1e-12);
        assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
    }
}
