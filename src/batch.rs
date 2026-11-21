//! Batch mode — run the same filter over many CSV files at once.
//!
//! `batch` takes a directory of `*.csv` files, filters each one with
//! identical parameters, and writes `<stem>.filtered.csv` +
//! `<stem>.summary.txt` next to each input (or into an output
//! directory). A failed file (bad CSV, too few samples) is reported
//! and skipped instead of aborting the whole run, and the final table
//! lists every file with its status so a bad apple is visible.

use crate::cv::CvKalman;
use crate::errors::{KalmanError::CsvParse, Result};
use crate::output::{cv_csv, scalar_csv, summarise_cv, summarise_scalar, RunSummary};
use crate::scalar::ScalarKalman;
use crate::series::parse_csv;
use std::path::{Path, PathBuf};

/// Which filter a batch run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchFilter {
    /// Scalar random-walk filter.
    Scalar,
    /// Constant-velocity filter.
    Cv,
}

/// Parameters for a batch run.
#[derive(Debug, Clone, Copy)]
pub struct BatchParams {
    /// Which filter to run.
    pub filter: BatchFilter,
    /// Process-noise variance `Q` (scalar) or PSD `q` (CV).
    pub q: f64,
    /// Measurement-noise variance `R`.
    pub r: f64,
    /// Initial error variance `P0`.
    pub p0: f64,
}

/// Result for one file in a batch run.
#[derive(Debug, Clone)]
pub struct BatchFileResult {
    /// Input file name (not the full path).
    pub file_name: String,
    /// `Ok` summary, or the error message when the file failed.
    pub outcome: std::result::Result<RunSummary, String>,
}

/// Discover `*.csv` files in a directory (sorted by name, non-recursive).
pub fn discover_csv_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir)?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .map(|ext| ext.eq_ignore_ascii_case("csv"))
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CsvParse(format!(
            "no .csv files found in {}",
            dir.display()
        )));
    }
    Ok(files)
}

/// Filter one series with the batch parameters. `dts` are the time
/// steps (used by the CV filter); for the scalar filter they are
/// ignored. Returns the output CSV text and the summary.
pub fn run_one(
    text: &str,
    params: BatchParams,
) -> Result<(String, RunSummary)> {
    let series = parse_csv(text)?;
    match params.filter {
        BatchFilter::Scalar => {
            let first = series.samples[0].z;
            let mut kf = ScalarKalman::new(first, params.p0, params.q, params.r)?;
            let steps: Vec<_> = series
                .measurements()
                .iter()
                .map(|&z| kf.step_plain(z))
                .collect();
            let summary = summarise_scalar(&series, &steps);
            Ok((scalar_csv(&series, &steps), summary))
        }
        BatchFilter::Cv => {
            let first = series.samples[0].z;
            let mut kf = CvKalman::new(first, 0.0, params.p0, params.q, params.r)?;
            let dts = series.dts();
            let mut steps = Vec::with_capacity(series.len());
            for (i, s) in series.samples.iter().enumerate() {
                // The first sample has dt = 0; give it a 1 s step so the
                // filter still performs one predict/update cycle.
                let dt = if i == 0 { 1.0 } else { dts[i].max(1e-9) };
                steps.push(kf.step(s.z, dt)?);
            }
            let summary = summarise_cv(&series, &steps);
            Ok((cv_csv(&series, &steps), summary))
        }
    }
}

/// Run a batch over every `*.csv` in `input_dir`, writing results to
/// `output_dir` (created when missing). Returns one result per file,
/// in file-name order.
pub fn run_batch(
    input_dir: &Path,
    output_dir: &Path,
    params: BatchParams,
) -> Result<Vec<BatchFileResult>> {
    let files = discover_csv_files(input_dir)?;
    std::fs::create_dir_all(output_dir)?;
    let mut results = Vec::with_capacity(files.len());
    for file in &files {
        let file_name = file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stem = file
            .file_stem()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let outcome = (|| -> Result<RunSummary> {
            let text = std::fs::read_to_string(file)?;
            let (csv, summary) = run_one(&text, params)?;
            std::fs::write(output_dir.join(format!("{stem}.filtered.csv")), csv)?;
            std::fs::write(
                output_dir.join(format!("{stem}.summary.txt")),
                summary.render(&format!("kalman-filter-1d — {file_name}")),
            )?;
            Ok(summary)
        })();
        results.push(BatchFileResult {
            file_name,
            outcome: outcome.map_err(|e| e.to_string()),
        });
    }
    Ok(results)
}

/// Render the batch table printed at the end of a batch run.
pub fn render_batch_table(results: &[BatchFileResult]) -> String {
    let mut out = String::from("file, samples, rmse_raw, rmse_filtered, improvement, nees_mean, status\n");
    for r in results {
        match &r.outcome {
            Ok(s) => {
                let raw = s
                    .rmse_raw
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".to_string());
                let filt = s
                    .rmse_filtered
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".to_string());
                let imp = s
                    .improvement
                    .map(|v| format!("{v:.2}x"))
                    .unwrap_or_else(|| "-".to_string());
                let nees = s
                    .nees
                    .map(|rep| format!("{:.3}", rep.mean))
                    .unwrap_or_else(|| "-".to_string());
                out.push_str(&format!(
                    "{}, {}, {raw}, {filt}, {imp}, {nees}, ok\n",
                    r.file_name, s.n
                ));
            }
            Err(msg) => {
                out.push_str(&format!("{}, -, -, -, -, -, FAILED: {msg}\n", r.file_name));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(dir: &Path, name: &str, text: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(text.as_bytes()).unwrap();
    }

    #[test]
    fn batch_processes_every_csv_and_skips_bad_ones() {
        let base = std::env::temp_dir().join("kf1d-batch-test");
        let _ = std::fs::remove_dir_all(&base);
        let input = base.join("in");
        let output = base.join("out");
        std::fs::create_dir_all(&input).unwrap();
        write_csv(
            &input,
            "good.csv",
            "t,z,truth\n0.0,1.0,1.0\n1.0,1.1,1.0\n2.0,0.9,1.0\n",
        );
        write_csv(&input, "bad.csv", "t,z\n0.0,not-a-number\n");
        let params = BatchParams {
            filter: BatchFilter::Scalar,
            q: 0.01,
            r: 0.1,
            p0: 1.0,
        };
        let results = run_batch(&input, &output, params).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].file_name, "bad.csv");
        assert!(results[0].outcome.is_err());
        assert_eq!(results[1].file_name, "good.csv");
        assert!(results[1].outcome.is_ok());
        assert!(output.join("good.filtered.csv").exists());
        assert!(output.join("good.summary.txt").exists());
        assert!(!output.join("bad.filtered.csv").exists());
        let table = render_batch_table(&results);
        assert!(table.contains("good.csv"));
        assert!(table.contains("FAILED"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn discover_finds_only_csv_files() {
        let base = std::env::temp_dir().join("kf1d-discover-test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        write_csv(&base, "a.csv", "t,z\n0.0,1.0\n");
        write_csv(&base, "b.CSV", "t,z\n0.0,1.0\n");
        write_csv(&base, "c.txt", "t,z\n0.0,1.0\n");
        let files = discover_csv_files(&base).unwrap();
        assert_eq!(files.len(), 2);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn run_one_cv_handles_a_series() {
        let text = "t,z,truth\n0.0,0.0,0.0\n1.0,0.1,0.1\n2.0,0.2,0.2\n";
        let params = BatchParams {
            filter: BatchFilter::Cv,
            q: 0.01,
            r: 0.1,
            p0: 1.0,
        };
        let (csv, summary) = run_one(text, params).unwrap();
        assert!(csv.contains("gain_x"));
        assert_eq!(summary.n, 3);
    }
}
