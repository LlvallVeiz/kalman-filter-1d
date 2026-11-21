//! Integration tests for CSV parsing and the batch runner, including
//! the CSV round-trip through the generator.

use kalman_filter_1d::batch::{discover_csv_files, run_batch, BatchFilter, BatchParams};
use kalman_filter_1d::series::{parse_csv, to_csv, TimeSeries};
use kalman_filter_1d::synth::{generate, Scenario, SynthParams};
use std::io::Write;

#[test]
fn generated_series_round_trips_through_csv() {
    let series = generate(&SynthParams::defaults(Scenario::Sine)).unwrap();
    let reparsed = parse_csv(&to_csv(&series)).unwrap();
    assert_eq!(series, reparsed);
}

#[test]
fn irregular_sampling_gives_the_right_dts() {
    let ts = parse_csv("t,z,truth\n0.0,1.0,1.0\n0.5,1.1,1.0\n2.0,0.9,1.0\n").unwrap();
    assert_eq!(ts.dts(), vec![0.0, 0.5, 1.5]);
}

#[test]
fn batch_cv_mode_writes_filtered_files() {
    let base = std::env::temp_dir().join("kf1d-batch-cv-test");
    let _ = std::fs::remove_dir_all(&base);
    let input = base.join("in");
    let output = base.join("out");
    std::fs::create_dir_all(&input).unwrap();
    let series = generate(&SynthParams::defaults(Scenario::Ramp)).unwrap();
    let mut f = std::fs::File::create(input.join("ramp.csv")).unwrap();
    f.write_all(to_csv(&series).as_bytes()).unwrap();
    let params = BatchParams {
        filter: BatchFilter::Cv,
        q: 0.01,
        r: 0.25,
        p0: 1.0,
    };
    let results = run_batch(&input, &output, params).unwrap();
    assert_eq!(results.len(), 1);
    let summary = results[0].outcome.as_ref().unwrap();
    assert!(summary.rmse_filtered.unwrap() < summary.rmse_raw.unwrap());
    assert!(output.join("ramp.filtered.csv").exists());
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn discover_rejects_an_empty_directory() {
    let base = std::env::temp_dir().join("kf1d-batch-empty-test");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    assert!(discover_csv_files(&base).is_err());
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn time_series_rejects_empty_input() {
    assert!(TimeSeries::new(vec![]).is_err());
}
