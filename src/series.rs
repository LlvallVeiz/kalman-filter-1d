//! Time-series data model and CSV parsing.
//!
//! Input format (one header line, then one sample per line):
//!
//! ```text
//! t,z,truth
//! 0.0,1.02,1.0
//! 1.0,0.98,1.0
//! ```
//!
//! * `t` — time (seconds); must be non-decreasing. The CV filter uses
//!   the *differences* as its `dt`, so irregular sampling is supported.
//! * `z` — measurement.
//! * `truth` — optional ground truth (third column). When present, the
//!   CLI reports RMSE and NEES; when absent it reports only the
//!   filtered series. A missing third column, an empty field, or the
//!   literal `NA` all mean "no ground truth for this sample".
//!
//! A header is required and must name the first two columns `t` and
//! `z` (case-insensitive); the third column, when present, must be
//! named `truth`. Blank lines and lines starting with `#` are ignored.

use crate::errors::{KalmanError::CsvParse, KalmanError::InsufficientData, Result};

/// One sample of a measured time series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Time in seconds.
    pub t: f64,
    /// Measurement.
    pub z: f64,
    /// Ground truth, when known.
    pub truth: Option<f64>,
}

/// A measured time series: samples in non-decreasing time order.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSeries {
    /// Samples in time order.
    pub samples: Vec<Sample>,
}

impl TimeSeries {
    /// Build from samples, verifying non-decreasing time.
    pub fn new(samples: Vec<Sample>) -> Result<Self> {
        if samples.is_empty() {
            return Err(InsufficientData("time series is empty".into()));
        }
        for w in samples.windows(2) {
            if w[1].t < w[0].t {
                return Err(CsvParse(format!(
                    "times must be non-decreasing, got {} after {}",
                    w[1].t, w[0].t
                )));
            }
        }
        Ok(Self { samples })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the series is empty (never true for a validated series).
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Time steps between consecutive samples (`dt[i]` is the step
    /// *into* sample `i`; `dt[0] = 0`).
    pub fn dts(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.samples.len()];
        for (i, pair) in self.samples.windows(2).enumerate() {
            out[i + 1] = pair[1].t - pair[0].t;
        }
        out
    }

    /// Measurements only.
    pub fn measurements(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.z).collect()
    }

    /// Ground truth where present (`None` entries where absent).
    pub fn truths(&self) -> Vec<Option<f64>> {
        self.samples.iter().map(|s| s.truth).collect()
    }

    /// `true` when *every* sample carries ground truth.
    pub fn has_full_truth(&self) -> bool {
        self.samples.iter().all(|s| s.truth.is_some())
    }
}

/// Parse a CSV document in the format described above.
/// Leading blank lines and `#` comment lines before the header are
/// skipped, so a commented file still parses.
pub fn parse_csv(text: &str) -> Result<TimeSeries> {
    let mut lines = text.lines();
    // Find the header: skip leading blank / comment lines.
    let header = loop {
        match lines.next() {
            Some(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                break line;
            }
            None => return Err(CsvParse("empty CSV document".into())),
        }
    };
    let cols: Vec<&str> = header.split(',').map(str::trim).collect();
    if cols.len() < 2 || cols.len() > 3 {
        return Err(CsvParse(format!(
            "header must have 2 or 3 columns (t,z[,truth]), got {}: {header}",
            cols.len()
        )));
    }
    if !cols[0].eq_ignore_ascii_case("t") || !cols[1].eq_ignore_ascii_case("z") {
        return Err(CsvParse(format!(
            "first two columns must be t,z, got: {header}"
        )));
    }
    let has_truth_col = cols.len() == 3;
    if has_truth_col && !cols[2].eq_ignore_ascii_case("truth") {
        return Err(CsvParse(format!(
            "third column must be named truth, got: {header}"
        )));
    }

    let mut samples = Vec::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != cols.len() {
            return Err(CsvParse(format!(
                "line {}: expected {} fields, got {}: {raw}",
                line_no + 2,
                cols.len(),
                fields.len()
            )));
        }
        let t = parse_number(fields[0], "t", line_no + 2)?;
        let z = parse_number(fields[1], "z", line_no + 2)?;
        let truth = if has_truth_col {
            parse_optional_number(fields[2], line_no + 2)?
        } else {
            None
        };
        samples.push(Sample { t, z, truth });
    }
    TimeSeries::new(samples)
}

fn parse_number(field: &str, name: &str, line_no: usize) -> Result<f64> {
    field.parse::<f64>().map_err(|_| {
        CsvParse(format!(
            "line {line_no}: cannot parse {name} = {field:?} as a number"
        ))
    })
}

fn parse_optional_number(field: &str, line_no: usize) -> Result<Option<f64>> {
    if field.is_empty() || field.eq_ignore_ascii_case("na") {
        return Ok(None);
    }
    field.parse::<f64>().map(Some).map_err(|_| {
        CsvParse(format!(
            "line {line_no}: cannot parse truth = {field:?} as a number"
        ))
    })
}

/// Serialise a series back to CSV (used by the generator and by tests).
pub fn to_csv(series: &TimeSeries) -> String {
    let mut out = String::from("t,z,truth\n");
    for s in &series.samples {
        match s.truth {
            Some(truth) => out.push_str(&format!("{},{},{}\n", s.t, s.z, truth)),
            None => out.push_str(&format!("{},{},\n", s.t, s.z)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = "t,z,truth\n0.0,1.02,1.0\n1.0,0.98,1.0\n2.0,1.05,1.0\n";

    #[test]
    fn parses_three_column_csv() {
        let ts = parse_csv(SIMPLE).unwrap();
        assert_eq!(ts.len(), 3);
        assert_eq!(ts.samples[1], Sample { t: 1.0, z: 0.98, truth: Some(1.0) });
        assert_eq!(ts.dts(), vec![0.0, 1.0, 1.0]);
        assert!(ts.has_full_truth());
    }

    #[test]
    fn parses_two_column_csv_without_truth() {
        let ts = parse_csv("t,z\n0.0,1.02\n1.0,0.98\n").unwrap();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.samples[0].truth, None);
        assert!(!ts.has_full_truth());
    }

    #[test]
    fn empty_truth_field_and_na_mean_missing() {
        let ts = parse_csv("t,z,truth\n0.0,1.0,\n1.0,2.0,NA\n2.0,3.0,3.0\n").unwrap();
        assert_eq!(ts.truths(), vec![None, None, Some(3.0)]);
        assert!(!ts.has_full_truth());
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let ts = parse_csv("# a comment\nt,z,truth\n0.0,1.0,1.0\n\n# another\n1.0,2.0,2.0\n").unwrap();
        assert_eq!(ts.len(), 2);
    }

    #[test]
    fn rejects_decreasing_time() {
        let err = parse_csv("t,z\n1.0,1.0\n0.0,2.0\n").unwrap_err();
        assert!(err.to_string().contains("non-decreasing"));
    }

    #[test]
    fn rejects_bad_header_and_bad_numbers() {
        assert!(parse_csv("time,value\n0.0,1.0\n").is_err());
        assert!(parse_csv("t,z\n0.0,abc\n").is_err());
        assert!(parse_csv("t,z\n0.0\n").is_err());
        assert!(parse_csv("").is_err());
        assert!(parse_csv("t,z\n").is_err());
    }

    #[test]
    fn round_trip_through_csv() {
        let ts = parse_csv(SIMPLE).unwrap();
        let back = parse_csv(&to_csv(&ts)).unwrap();
        assert_eq!(ts, back);
    }
}
