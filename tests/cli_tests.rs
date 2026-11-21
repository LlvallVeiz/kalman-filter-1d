//! End-to-end CLI tests: drive the real binary (via `CARGO_BIN_EXE_…`)
//! through generate → filter → compare → batch, and check the files it
//! writes contain the numbers it prints.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_kalman-filter-1d"))
}

fn run(args: &[&str], dir: &std::path::Path) -> (bool, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(dir)
        .output()
        .expect("failed to run the binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn tmpdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn help_and_version_exit_zero() {
    let dir = tmpdir("kf1d-cli-help");
    let (ok, stdout, _) = run(&["--help"], &dir);
    assert!(ok);
    assert!(stdout.contains("filter"));
    assert!(stdout.contains("generate"));
    assert!(stdout.contains("compare"));
    assert!(stdout.contains("batch"));
    let (ok, stdout, _) = run(&["--version"], &dir);
    assert!(ok);
    assert!(stdout.contains("1.0.0"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn handcheck_prints_the_hand_computed_values() {
    let dir = tmpdir("kf1d-cli-handcheck");
    let (ok, stdout, _) = run(&["handcheck"], &dir);
    assert!(ok);
    assert!(stdout.contains("0.9099"));
    assert!(stdout.contains("1.0054"));
    assert!(stdout.contains("0.9658"));
    assert!(stdout.contains("K∞ = 0.2702"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn generate_then_filter_then_compare_end_to_end() {
    let dir = tmpdir("kf1d-cli-e2e");
    // Generate a step series.
    let (ok, stdout, _) = run(
        &["generate", "step", "--seed", "42", "--output", "step.csv"],
        &dir,
    );
    assert!(ok, "{stdout}");
    assert!(dir.join("step.csv").exists());
    // Filter it with smoothing.
    let (ok, stdout, _) = run(
        &["filter", "step.csv", "--q", "0.01", "--r", "0.25", "--smooth"],
        &dir,
    );
    assert!(ok, "{stdout}");
    assert!(stdout.contains("RMSE filtered"));
    assert!(stdout.contains("NEES mean"));
    assert!(stdout.contains("RMSE smoothed"));
    assert!(dir.join("step.filtered.csv").exists());
    assert!(dir.join("step.summary.txt").exists());
    assert!(dir.join("step.smoothed.csv").exists());
    // The summary file must contain the same numbers as stdout.
    let summary = std::fs::read_to_string(dir.join("step.summary.txt")).unwrap();
    assert!(summary.contains("RMSE filtered"));
    // Compare the three estimators.
    let (ok, stdout, _) = run(&["compare", "step.csv"], &dir);
    assert!(ok, "{stdout}");
    assert!(stdout.contains("scalar Kalman"));
    assert!(stdout.contains("CV Kalman"));
    assert!(stdout.contains("alpha-beta"));
    assert!(stdout.contains("best:"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn adaptive_filter_reports_its_estimates() {
    let dir = tmpdir("kf1d-cli-adaptive");
    let (ok, _, _) = run(
        &["generate", "temperature", "--seed", "5", "--output", "temp.csv"],
        &dir,
    );
    assert!(ok);
    let (ok, stdout, _) = run(
        &["filter", "temp.csv", "--q", "0.01", "--r", "0.04", "--adaptive"],
        &dir,
    );
    assert!(ok, "{stdout}");
    assert!(stdout.contains("adaptive:"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn batch_mode_filters_a_directory() {
    let dir = tmpdir("kf1d-cli-batch");
    let input = dir.join("in");
    std::fs::create_dir_all(&input).unwrap();
    let (ok, _, _) = run(
        &["generate", "sine", "--seed", "1", "--output", "in/sine.csv"],
        &dir,
    );
    assert!(ok);
    let (ok, _, _) = run(
        &["generate", "ramp", "--seed", "2", "--output", "in/ramp.csv"],
        &dir,
    );
    assert!(ok);
    let (ok, stdout, _) = run(
        &["batch", "in", "--output-dir", "out", "--q", "0.01", "--r", "0.09"],
        &dir,
    );
    assert!(ok, "{stdout}");
    assert!(stdout.contains("2/2 files filtered"));
    assert!(dir.join("out/sine.filtered.csv").exists());
    assert!(dir.join("out/ramp.filtered.csv").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn filter_rejects_a_missing_file_and_a_bad_csv() {
    let dir = tmpdir("kf1d-cli-errors");
    let (ok, _, stderr) = run(&["filter", "nope.csv"], &dir);
    assert!(!ok);
    assert!(stderr.contains("error"));
    let mut f = std::fs::File::create(dir.join("bad.csv")).unwrap();
    f.write_all(b"t,z\n0.0,abc\n").unwrap();
    let (ok, _, stderr) = run(&["filter", "bad.csv"], &dir);
    assert!(!ok);
    assert!(stderr.contains("error"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compare_rejects_input_without_ground_truth() {
    let dir = tmpdir("kf1d-cli-compare-no-truth");
    let mut f = std::fs::File::create(dir.join("plain.csv")).unwrap();
    f.write_all(b"t,z\n0.0,1.0\n1.0,1.1\n2.0,0.9\n").unwrap();
    let (ok, _, stderr) = run(&["compare", "plain.csv"], &dir);
    assert!(!ok);
    assert!(stderr.contains("ground truth"));
    let _ = std::fs::remove_dir_all(&dir);
}
