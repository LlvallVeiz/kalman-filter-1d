//! Output: filtered CSV files and human-readable summaries.
//!
//! Filtered CSV columns (scalar):
//!
//! ```text
//! t,z,truth,x,p,gain,innovation
//! ```
//!
//! CV output adds `v,p_v` (velocity estimate and its variance).
//! Smoothed output adds `x_smooth,p_smooth` instead of the filter's
//! `x,p` (the filter columns are kept for comparison).
//!
//! The summary is a short text block (also written to `<stem>.summary.txt`
//! next to the CSV by the CLI) with the RMSE of the raw measurements,
//! the RMSE of the filter, the improvement factor, and — when ground
//! truth exists — the NEES consistency verdict.

use crate::cv::CvStep;
use crate::metrics::{nees_cv, nees_report, nees_scalar, rmse, NeesReport};
use crate::scalar::ScalarStep;
use crate::series::TimeSeries;

/// Summary statistics for one filtered run.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Number of samples.
    pub n: usize,
    /// RMSE of the raw measurements vs ground truth (None without truth).
    pub rmse_raw: Option<f64>,
    /// RMSE of the filtered estimates vs ground truth.
    pub rmse_filtered: Option<f64>,
    /// RMSE of the smoothed estimates vs ground truth (smoother runs).
    pub rmse_smoothed: Option<f64>,
    /// Improvement factor `rmse_raw / rmse_filtered` (None without truth).
    pub improvement: Option<f64>,
    /// NEES consistency report (None without truth).
    pub nees: Option<NeesReport>,
    /// Mean NIS over the run (innovation consistency, always available).
    pub mean_nis: f64,
}

impl RunSummary {
    /// Render the summary as the text block the CLI prints and saves.
    pub fn render(&self, title: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("{title}\n"));
        s.push_str(&format!("samples: {}\n", self.n));
        match (self.rmse_raw, self.rmse_filtered) {
            (Some(raw), Some(filt)) => {
                s.push_str(&format!("RMSE raw measurements : {raw:.6}\n"));
                s.push_str(&format!("RMSE filtered         : {filt:.6}\n"));
                if let Some(imp) = self.improvement {
                    s.push_str(&format!("improvement factor    : {imp:.3}x\n"));
                }
                if let Some(sm) = self.rmse_smoothed {
                    s.push_str(&format!("RMSE smoothed         : {sm:.6}\n"));
                }
            }
            _ => {
                s.push_str("RMSE: no ground truth in input, skipped\n");
                if let Some(sm) = self.rmse_smoothed {
                    s.push_str(&format!("RMSE smoothed         : {sm:.6}\n"));
                }
            }
        }
        match &self.nees {
            Some(rep) => {
                s.push_str(&format!(
                    "NEES mean             : {:.4} (95% interval [{:.4}, {:.4}]) → {}\n",
                    rep.mean,
                    rep.lower_95,
                    rep.upper_95,
                    if rep.consistent_95 {
                        "CONSISTENT"
                    } else {
                        "INCONSISTENT"
                    }
                ));
            }
            None => {
                s.push_str("NEES: no ground truth in input, skipped\n");
            }
        }
        s.push_str(&format!("mean NIS              : {:.4}\n", self.mean_nis));
        s
    }
}

/// Build a [`RunSummary`] for a scalar run from the filter steps and
/// the original series.
pub fn summarise_scalar(series: &TimeSeries, steps: &[ScalarStep]) -> RunSummary {
    let estimates: Vec<f64> = steps.iter().map(|s| s.x).collect();
    let mean_nis = steps
        .iter()
        .map(|s| s.innovation * s.innovation / s.innovation_variance)
        .sum::<f64>()
        / steps.len().max(1) as f64;
    let mut summary = RunSummary {
        n: series.len(),
        rmse_raw: None,
        rmse_filtered: None,
        rmse_smoothed: None,
        improvement: None,
        nees: None,
        mean_nis,
    };
    if series.has_full_truth() {
        let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
        let raw = series.measurements();
        if let Ok(r) = rmse(&raw, &truth) {
            summary.rmse_raw = Some(r);
        }
        if let Ok(f) = rmse(&estimates, &truth) {
            summary.rmse_filtered = Some(f);
            if let Some(r) = summary.rmse_raw {
                if f > 0.0 {
                    summary.improvement = Some(r / f);
                }
            }
        }
        let nees: Vec<f64> = steps
            .iter()
            .zip(truth.iter())
            .filter_map(|(s, t)| nees_scalar(s.x, *t, s.p))
            .collect();
        if let Ok(rep) = nees_report(&nees, 1) {
            summary.nees = Some(rep);
        }
    }
    summary
}

/// Build a [`RunSummary`] for a CV run (position RMSE/NEES only;
/// velocity truth is not part of the CSV format).
pub fn summarise_cv(series: &TimeSeries, steps: &[CvStep]) -> RunSummary {
    let estimates: Vec<f64> = steps.iter().map(|s| s.x[0]).collect();
    let mean_nis = steps.iter().map(|s| s.nis).sum::<f64>() / steps.len().max(1) as f64;
    let mut summary = RunSummary {
        n: series.len(),
        rmse_raw: None,
        rmse_filtered: None,
        rmse_smoothed: None,
        improvement: None,
        nees: None,
        mean_nis,
    };
    if series.has_full_truth() {
        let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
        let raw = series.measurements();
        if let Ok(r) = rmse(&raw, &truth) {
            summary.rmse_raw = Some(r);
        }
        if let Ok(f) = rmse(&estimates, &truth) {
            summary.rmse_filtered = Some(f);
            if let Some(r) = summary.rmse_raw {
                if f > 0.0 {
                    summary.improvement = Some(r / f);
                }
            }
        }
        // CV truth for velocity is unknown, so NEES uses position only
        // with the marginal variance P₀₀ — a 1-dof test.
        let nees: Vec<f64> = steps
            .iter()
            .zip(truth.iter())
            .filter_map(|(s, t)| nees_scalar(s.x[0], *t, s.p[0][0]))
            .collect();
        if let Ok(rep) = nees_report(&nees, 1) {
            summary.nees = Some(rep);
        }
    }
    summary
}

/// NEES of a full CV run against position+velocity truth (used by
/// tests and by the `compare` command when velocity truth is known —
/// currently only the synthetic altitude scenario supplies it via the
/// generator API, not via CSV).
pub fn nees_cv_full(steps: &[CvStep], truth: &[[f64; 2]]) -> Option<NeesReport> {
    let values: Vec<f64> = steps
        .iter()
        .zip(truth.iter())
        .filter_map(|(s, t)| nees_cv(s.x, *t, s.p))
        .collect();
    nees_report(&values, 2).ok()
}

/// Serialise scalar filter output to CSV.
pub fn scalar_csv(series: &TimeSeries, steps: &[ScalarStep]) -> String {
    let mut out = String::from("t,z,truth,x,p,gain,innovation\n");
    for (s, st) in series.samples.iter().zip(steps.iter()) {
        match s.truth {
            Some(truth) => out.push_str(&format!(
                "{},{},{},{:.10},{:.10},{:.10},{:.10}\n",
                s.t, s.z, truth, st.x, st.p, st.gain, st.innovation
            )),
            None => out.push_str(&format!(
                "{},{},,{:.10},{:.10},{:.10},{:.10}\n",
                s.t, s.z, st.x, st.p, st.gain, st.innovation
            )),
        }
    }
    out
}

/// Serialise CV filter output to CSV.
pub fn cv_csv(series: &TimeSeries, steps: &[CvStep]) -> String {
    let mut out = String::from("t,z,truth,x,v,p,p_v,gain_x,gain_v,innovation\n");
    for (s, st) in series.samples.iter().zip(steps.iter()) {
        match s.truth {
            Some(truth) => out.push_str(&format!(
                "{},{},{},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}\n",
                s.t,
                s.z,
                truth,
                st.x[0],
                st.x[1],
                st.p[0][0],
                st.p[1][1],
                st.gain[0],
                st.gain[1],
                st.innovation
            )),
            None => out.push_str(&format!(
                "{},{},,{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}\n",
                s.t,
                s.z,
                st.x[0],
                st.x[1],
                st.p[0][0],
                st.p[1][1],
                st.gain[0],
                st.gain[1],
                st.innovation
            )),
        }
    }
    out
}

/// Serialise scalar smoother output to CSV (filter columns kept, plus
/// the smoothed estimate and variance).
pub fn scalar_smooth_csv(
    series: &TimeSeries,
    steps: &[ScalarStep],
    smoothed: &[crate::smoother::SmoothedScalar],
) -> String {
    let mut out = String::from("t,z,truth,x,p,x_smooth,p_smooth\n");
    for ((s, st), sm) in series.samples.iter().zip(steps.iter()).zip(smoothed.iter()) {
        match s.truth {
            Some(truth) => out.push_str(&format!(
                "{},{},{},{:.10},{:.10},{:.10},{:.10}\n",
                s.t, s.z, truth, st.x, st.p, sm.x, sm.p
            )),
            None => out.push_str(&format!(
                "{},{},,{:.10},{:.10},{:.10},{:.10}\n",
                s.t, s.z, st.x, st.p, sm.x, sm.p
            )),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::ScalarKalman;

    #[test]
    fn summary_shows_filter_beats_raw_on_slow_temperature_signal() {
        // The temperature scenario moves slowly (diurnal swing + drift),
        // which is what the scalar random-walk filter is built for:
        // it must beat the raw measurements by a wide margin.
        let params =
            crate::synth::SynthParams::defaults(crate::synth::Scenario::Temperature);
        let series = crate::synth::generate(&params).unwrap();
        let mut kf = ScalarKalman::new(25.0, 1.0, 0.01, 0.04).unwrap();
        let steps: Vec<_> = series
            .measurements()
            .iter()
            .map(|&z| kf.step_plain(z))
            .collect();
        let summary = summarise_scalar(&series, &steps);
        let raw = summary.rmse_raw.unwrap();
        let filt = summary.rmse_filtered.unwrap();
        assert!(filt < raw, "filter RMSE {filt} should beat raw {raw}");
        assert!(
            summary.improvement.unwrap() > 1.5,
            "improvement {:?} too small (raw {raw}, filtered {filt})",
            summary.improvement
        );
        let rendered = summary.render("test");
        assert!(rendered.contains("RMSE filtered"));
        assert!(rendered.contains("NEES mean"));
    }

    #[test]
    fn scalar_csv_has_one_row_per_sample() {
        let series = crate::synth::generate(
            &crate::synth::SynthParams::defaults(crate::synth::Scenario::Ramp),
        )
        .unwrap();
        let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.25).unwrap();
        let steps: Vec<_> = series
            .measurements()
            .iter()
            .map(|&z| kf.step_plain(z))
            .collect();
        let csv = scalar_csv(&series, &steps);
        assert_eq!(csv.lines().count(), series.len() + 1);
        assert!(csv.starts_with("t,z,truth,x,p,gain,innovation\n"));
    }
}
