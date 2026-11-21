//! `kalman-filter-1d` — one-dimensional Kalman filtering in pure Rust.
//!
//! A std-only library (zero external crates) for estimating a hidden
//! one-dimensional state from noisy measurements:
//!
//! * [`scalar`] — the classic scalar Kalman filter (random-walk model).
//! * [`cv`] — the constant-velocity filter: position + velocity states.
//! * [`alpha_beta`] — the fixed-gain α–β tracker, the standard baseline
//!   for comparison (Benedict & Bordner, 1962; Kalata, 1984).
//! * [`smoother`] — Rauch–Tung–Striebel fixed-interval smoothing.
//! * [`noise`] — adaptive estimation of `Q` and `R` from the
//!   innovations (Mehra, 1970).
//! * [`synth`] — reproducible synthetic scenarios with ground truth
//!   (step, sine, ramp, temperature sensor, altitude random walk).
//! * [`metrics`] — RMSE and NEES consistency testing with exact
//!   chi-square quantiles.
//! * [`series`] / [`output`] / [`batch`] — CSV I/O and batch mode.
//!
//! The binary (`kalman-filter-1d`) exposes the same functionality as a
//! CLI; see `cli::USAGE` or run with `--help`.
//!
//! # Quick example
//!
//! ```rust
//! use kalman_filter_1d::scalar::ScalarKalman;
//!
//! let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
//! for z in [1.0, 1.1, 0.9] {
//!     let step = kf.step_plain(z);
//!     println!("x = {:.4}, P = {:.4}, K = {:.4}", step.x, step.p, step.gain);
//! }
//! // x = 0.9099, P = 0.0910, K = 0.9099
//! // x = 1.0054, P = 0.0502, K = 0.5025
//! // x = 0.9658, P = 0.0376, K = 0.3760
//! ```

pub mod alpha_beta;
pub mod batch;
pub mod cli;
pub mod cv;
pub mod errors;
pub mod metrics;
pub mod noise;
pub mod output;
pub mod rng;
pub mod scalar;
pub mod series;
pub mod smoother;
pub mod synth;
