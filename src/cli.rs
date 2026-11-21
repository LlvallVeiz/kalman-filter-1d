//! Command-line interface (std-only argument parsing).
//!
//! Subcommands:
//!
//! ```text
//! kalman-filter-1d filter   <input.csv> [options]   run a filter, write CSV + summary
//! kalman-filter-1d generate <scenario>  [options]   write a synthetic series
//! kalman-filter-1d compare  <input.csv> [options]   scalar KF vs CV KF vs alpha-beta
//! kalman-filter-1d batch    <input-dir> [options]   filter every CSV in a directory
//! kalman-filter-1d handcheck                  print the hand-checkable example
//! kalman-filter-1d --help / --version
//! ```
//!
//! Common filter options: `--filter scalar|cv`, `--q`, `--r`, `--p0`,
//! `--x0`, `--output`, `--smooth`, `--adaptive [lambda]`.
//! The parser accepts both `--key value` and `--key=value`.
//!
//! Exit code is 0 on success, 1 on any error (the message goes to
//! stderr). No external argument-parsing crate is used so the project
//! builds fully offline.

use crate::alpha_beta::AlphaBeta;
use crate::batch::{run_batch, render_batch_table, BatchFilter, BatchParams};
use crate::cv::CvKalman;
use crate::errors::{KalmanError, KalmanError::Cli, Result};
use crate::metrics::rmse;
use crate::noise::InnovationTracker;
use crate::output::{
    cv_csv, scalar_csv, scalar_smooth_csv, summarise_cv, summarise_scalar,
};
use crate::scalar::{run_scalar_series, ScalarKalman};
use crate::series::{parse_csv, to_csv, TimeSeries};
use crate::smoother::{smooth_cv, smooth_scalar};
use crate::synth::{generate, parse_scenario, SynthParams};
use std::path::{Path, PathBuf};

/// Parsed command line.
#[derive(Debug)]
pub enum Command {
    /// `filter <input.csv> [options]`
    Filter(FilterOptions),
    /// `generate <scenario> [options]`
    Generate(GenerateOptions),
    /// `compare <input.csv> [options]`
    Compare(CompareOptions),
    /// `batch <input-dir> [options]`
    Batch(BatchOptions),
    /// `handcheck`
    Handcheck,
    /// `--help`
    Help,
    /// `--version`
    Version,
}

/// Options shared by `filter` (and partly by `compare` / `batch`).
#[derive(Debug, Clone)]
pub struct FilterOptions {
    /// Input CSV path.
    pub input: PathBuf,
    /// `scalar` (default) or `cv`.
    pub filter: String,
    /// Process-noise variance / PSD.
    pub q: f64,
    /// Measurement-noise variance.
    pub r: f64,
    /// Initial error variance.
    pub p0: f64,
    /// Initial state (default: first measurement).
    pub x0: Option<f64>,
    /// Output CSV path (default: `<stem>.filtered.csv` next to input).
    pub output: Option<PathBuf>,
    /// Also run the RTS smoother and write `.smoothed.csv`.
    pub smooth: bool,
    /// Adaptive noise estimation: `Some(lambda)` enables it.
    pub adaptive: Option<f64>,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            filter: "scalar".to_string(),
            q: 0.01,
            r: 0.1,
            p0: 1.0,
            x0: None,
            output: None,
            smooth: false,
            adaptive: None,
        }
    }
}

/// Options for `generate`.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Scenario name (`step`, `sine`, `ramp`, `temperature`, `altitude`).
    pub scenario: String,
    /// Number of samples.
    pub n: usize,
    /// Sampling interval in seconds.
    pub dt: f64,
    /// Measurement-noise standard deviation.
    pub noise_std: f64,
    /// RNG seed.
    pub seed: u64,
    /// Scenario parameter 1 (see [`SynthParams`]).
    pub p1: Option<f64>,
    /// Scenario parameter 2.
    pub p2: Option<f64>,
    /// Output CSV path (default: `<scenario>.csv` in the current directory).
    pub output: Option<PathBuf>,
}

/// Options for `compare`.
#[derive(Debug, Clone)]
pub struct CompareOptions {
    /// Input CSV path (must carry ground truth).
    pub input: PathBuf,
    /// Process-noise variance / PSD for both Kalman filters.
    pub q: f64,
    /// Measurement-noise variance for both Kalman filters.
    pub r: f64,
    /// Initial error variance.
    pub p0: f64,
    /// Alpha-beta gains: explicit `alpha`/`beta`, or a tracking index.
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub lambda: Option<f64>,
}

/// Options for `batch`.
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// Directory of input CSV files.
    pub input_dir: PathBuf,
    /// Output directory (default: `<input-dir>/filtered`).
    pub output_dir: Option<PathBuf>,
    /// `scalar` (default) or `cv`.
    pub filter: String,
    /// Process-noise variance / PSD.
    pub q: f64,
    /// Measurement-noise variance.
    pub r: f64,
    /// Initial error variance.
    pub p0: f64,
}

/// Parse `std::env::args()` into a [`Command`].
pub fn parse_args(args: &[String]) -> Result<Command> {
    if args.len() < 2 {
        return Ok(Command::Help);
    }
    match args[1].as_str() {
        "--help" | "-h" | "help" => Ok(Command::Help),
        "--version" | "-V" => Ok(Command::Version),
        "filter" => parse_filter(&args[2..]),
        "generate" => parse_generate(&args[2..]),
        "compare" => parse_compare(&args[2..]),
        "batch" => parse_batch(&args[2..]),
        "handcheck" => Ok(Command::Handcheck),
        other => Err(Cli(format!(
            "unknown subcommand {other:?}; run with --help for usage"
        ))),
    }
}

/// Parsed arguments: `(positionals, options)` where each option is a
/// key (without `--`) mapped to `Some(value)` or `None` for bare flags.
type ParsedArgs = (Vec<String>, Vec<(String, Option<String>)>);

/// Tiny `--key value` / `--key=value` / `--flag` parser.
///
/// Returns `(positionals, options)` where `options` maps a key
/// (without `--`) to `Some(value)` or `None` for bare flags.
fn split_args(args: &[String]) -> Result<ParsedArgs> {
    let mut positionals = Vec::new();
    let mut options = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if let Some(rest) = arg.strip_prefix("--") {
            if let Some((key, value)) = rest.split_once('=') {
                options.push((key.to_string(), Some(value.to_string())));
            } else if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                options.push((rest.to_string(), Some(args[i + 1].clone())));
                i += 1;
            } else {
                options.push((rest.to_string(), None));
            }
        } else if let Some(short) = arg.strip_prefix('-') {
            // Single-dash short flags: -h only (kept minimal on purpose).
            if short == "h" {
                options.push(("help".to_string(), None));
            } else {
                return Err(Cli(format!("unknown option -{short}")));
            }
        } else {
            positionals.push(arg.clone());
        }
        i += 1;
    }
    Ok((positionals, options))
}

fn opt_f64(options: &[(String, Option<String>)], key: &str) -> Result<Option<f64>> {
    for (k, v) in options {
        if k == key {
            let raw = v.clone().ok_or_else(|| {
                Cli(format!("--{key} needs a value"))
            })?;
            let parsed: f64 = raw.parse().map_err(|_| {
                Cli(format!("--{key} must be a number, got {raw:?}"))
            })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

fn opt_usize(options: &[(String, Option<String>)], key: &str) -> Result<Option<usize>> {
    for (k, v) in options {
        if k == key {
            let raw = v.clone().ok_or_else(|| Cli(format!("--{key} needs a value")))?;
            let parsed: usize = raw.parse().map_err(|_| {
                Cli(format!("--{key} must be a whole number, got {raw:?}"))
            })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

fn opt_u64(options: &[(String, Option<String>)], key: &str) -> Result<Option<u64>> {
    for (k, v) in options {
        if k == key {
            let raw = v.clone().ok_or_else(|| Cli(format!("--{key} needs a value")))?;
            let parsed: u64 = raw.parse().map_err(|_| {
                Cli(format!("--{key} must be a whole number, got {raw:?}"))
            })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

fn opt_string(options: &[(String, Option<String>)], key: &str) -> Option<String> {
    options
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.clone())
}

fn has_flag(options: &[(String, Option<String>)], key: &str) -> bool {
    options.iter().any(|(k, _)| k == key)
}

fn parse_filter(args: &[String]) -> Result<Command> {
    let (pos, opts) = split_args(args)?;
    if pos.is_empty() {
        return Err(Cli("filter needs an input CSV file".into()));
    }
    let mut o = FilterOptions {
        input: PathBuf::from(&pos[0]),
        ..Default::default()
    };
    if let Some(f) = opt_string(&opts, "filter") {
        if f != "scalar" && f != "cv" {
            return Err(Cli(format!(
                "--filter must be scalar or cv, got {f:?}"
            )));
        }
        o.filter = f;
    }
    if let Some(v) = opt_f64(&opts, "q")? {
        o.q = v;
    }
    if let Some(v) = opt_f64(&opts, "r")? {
        o.r = v;
    }
    if let Some(v) = opt_f64(&opts, "p0")? {
        o.p0 = v;
    }
    if let Some(v) = opt_f64(&opts, "x0")? {
        o.x0 = Some(v);
    }
    if let Some(v) = opt_string(&opts, "output") {
        o.output = Some(PathBuf::from(v));
    }
    o.smooth = has_flag(&opts, "smooth");
    if has_flag(&opts, "adaptive") {
        // `--adaptive` alone means lambda = 0.98; `--adaptive 0.95`
        // overrides it. A bare flag stores None as its value, so look
        // the raw entry up instead of going through opt_f64 (which
        // requires a value for every other numeric option).
        let lambda = match opts.iter().find(|(k, _)| k == "adaptive") {
            Some((_, Some(v))) => v.parse::<f64>().map_err(|_| {
                Cli(format!("--adaptive must be a number, got {v:?}"))
            })?,
            _ => 0.98,
        };
        o.adaptive = Some(lambda);
    }
    Ok(Command::Filter(o))
}

fn parse_generate(args: &[String]) -> Result<Command> {
    let (pos, opts) = split_args(args)?;
    if pos.is_empty() {
        return Err(Cli(
            "generate needs a scenario: step, sine, ramp, temperature or altitude".into(),
        ));
    }
    let scenario = parse_scenario(&pos[0])?;
    let mut defaults = SynthParams::defaults(scenario);
    if let Some(v) = opt_usize(&opts, "n")? {
        defaults.n = v;
    }
    if let Some(v) = opt_f64(&opts, "dt")? {
        defaults.dt = v;
    }
    if let Some(v) = opt_f64(&opts, "noise")? {
        defaults.noise_std = v;
    }
    if let Some(v) = opt_u64(&opts, "seed")? {
        defaults.seed = v;
    }
    Ok(Command::Generate(GenerateOptions {
        scenario: pos[0].clone(),
        n: defaults.n,
        dt: defaults.dt,
        noise_std: defaults.noise_std,
        seed: defaults.seed,
        p1: opt_f64(&opts, "p1")?,
        p2: opt_f64(&opts, "p2")?,
        output: opt_string(&opts, "output").map(PathBuf::from),
    }))
}

fn parse_compare(args: &[String]) -> Result<Command> {
    let (pos, opts) = split_args(args)?;
    if pos.is_empty() {
        return Err(Cli("compare needs an input CSV file".into()));
    }
    Ok(Command::Compare(CompareOptions {
        input: PathBuf::from(&pos[0]),
        q: opt_f64(&opts, "q")?.unwrap_or(0.01),
        r: opt_f64(&opts, "r")?.unwrap_or(0.1),
        p0: opt_f64(&opts, "p0")?.unwrap_or(1.0),
        alpha: opt_f64(&opts, "alpha")?,
        beta: opt_f64(&opts, "beta")?,
        lambda: opt_f64(&opts, "lambda")?,
    }))
}

fn parse_batch(args: &[String]) -> Result<Command> {
    let (pos, opts) = split_args(args)?;
    if pos.is_empty() {
        return Err(Cli("batch needs an input directory".into()));
    }
    let filter = opt_string(&opts, "filter").unwrap_or_else(|| "scalar".to_string());
    if filter != "scalar" && filter != "cv" {
        return Err(Cli(format!(
            "--filter must be scalar or cv, got {filter:?}"
        )));
    }
    Ok(Command::Batch(BatchOptions {
        input_dir: PathBuf::from(&pos[0]),
        output_dir: opt_string(&opts, "output-dir").map(PathBuf::from),
        filter,
        q: opt_f64(&opts, "q")?.unwrap_or(0.01),
        r: opt_f64(&opts, "r")?.unwrap_or(0.1),
        p0: opt_f64(&opts, "p0")?.unwrap_or(1.0),
    }))
}

/// Usage text printed by `--help`.
pub const USAGE: &str = "kalman-filter-1d 1.0.0 — one-dimensional Kalman filtering (pure Rust, std-only)

USAGE:
    kalman-filter-1d <subcommand> [options]

SUBCOMMANDS:
    filter <input.csv>        Run a filter over a CSV series (t,z[,truth]).
                              Writes <stem>.filtered.csv + <stem>.summary.txt
                              next to the input unless --output is given.
        --filter scalar|cv    Filter type (default: scalar)
        --q Q                 Process-noise variance / PSD (default: 0.01)
        --r R                 Measurement-noise variance (default: 0.1)
        --p0 P0               Initial error variance (default: 1.0)
        --x0 X0               Initial state (default: first measurement)
        --output FILE         Output CSV path
        --smooth              Also run the RTS smoother (writes .smoothed.csv)
        --adaptive [LAMBDA]   Retune Q/R from innovations (default lambda 0.98)

    generate <scenario>       Write a synthetic series with ground truth.
        Scenarios: step, sine, ramp, temperature, altitude
        --n N                 Number of samples (default: scenario default)
        --dt DT               Sampling interval in seconds
        --noise S             Measurement-noise std dev
        --seed S              RNG seed (default: 42)
        --p1 X --p2 Y         Scenario parameters (see README)
        --output FILE         Output CSV path (default: <scenario>.csv)

    compare <input.csv>       Scalar KF vs CV KF vs alpha-beta, side by side.
                              Needs ground truth in the input CSV.
        --q Q --r R --p0 P0   Noise parameters (defaults: 0.01, 0.1, 1.0)
        --alpha A --beta B    Alpha-beta gains (default: from --lambda 1.0)
        --lambda L            Alpha-beta tracking index (default: 1.0)

    batch <input-dir>         Filter every *.csv in a directory.
        --output-dir DIR      Output directory (default: <input-dir>/filtered)
        --filter scalar|cv    Filter type (default: scalar)
        --q Q --r R --p0 P0   Noise parameters

    handcheck                 Print the hand-checkable 3-step example.

    --help, -h                Show this help
    --version, -V             Show the version

EXAMPLES:
    kalman-filter-1d generate step --seed 42 --output examples/step.csv
    kalman-filter-1d filter examples/step.csv --q 0.01 --r 0.25 --smooth
    kalman-filter-1d compare examples/step.csv
    kalman-filter-1d batch examples --output-dir examples/filtered
";

/// Execute a parsed command. Returns the text to print on stdout.
pub fn run_command(cmd: Command) -> Result<String> {
    match cmd {
        Command::Help => Ok(USAGE.to_string()),
        Command::Version => Ok("kalman-filter-1d 1.0.0\n".to_string()),
        Command::Handcheck => Ok(handcheck_text()),
        Command::Filter(o) => run_filter(o),
        Command::Generate(o) => run_generate(o),
        Command::Compare(o) => run_compare(o),
        Command::Batch(o) => run_batch_cmd(o),
    }
}

fn read_series(path: &Path) -> Result<TimeSeries> {
    let text = std::fs::read_to_string(path)?;
    parse_csv(&text)
}

fn default_output_path(input: &Path, suffix: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{stem}.{suffix}"))
}

/// Summary path next to an output CSV: `<stem>.summary.txt`,
/// regardless of the output file's own name. A default filter output
/// `step.filtered.csv` gets `step.summary.txt` (not
/// `step.filtered.summary.txt`); a custom `--output custom.csv` gets
/// `custom.summary.txt`.
fn summary_path_for(out_path: &Path) -> PathBuf {
    let parent = out_path.parent().unwrap_or_else(|| Path::new("."));
    let name = out_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = if let Some(base) = name.strip_suffix(".filtered.csv") {
        base.to_string()
    } else if let Some(base) = name.strip_suffix(".csv") {
        base.to_string()
    } else {
        out_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".to_string())
    };
    parent.join(format!("{stem}.summary.txt"))
}

fn run_filter(o: FilterOptions) -> Result<String> {
    let series = read_series(&o.input)?;
    let mut out = String::new();
    match o.filter.as_str() {
        "scalar" => {
            let x0 = o.x0.unwrap_or(series.samples[0].z);
            let mut kf = ScalarKalman::new(x0, o.p0, o.q, o.r)?;
            let mut steps = Vec::with_capacity(series.len());
            let mut tracker = match o.adaptive {
                Some(lambda) => Some(InnovationTracker::new(lambda)?),
                None => None,
            };
            // Bounds for the adaptive retuning: the estimates are
            // damped (half old, half new) and clamped to a decade
            // around the starting values, so one bad window can
            // neither collapse R to zero nor explode Q. Without the
            // damping the R̂/Q feedback diverges (R̂ → 0, P → 0,
            // NEES → ∞) — see docs/RESEARCH.md §5.
            let (q0, r0) = (o.q, o.r);
            for (i, s) in series.samples.iter().enumerate() {
                let step = kf.step_plain(s.z);
                if let Some(t) = tracker.as_mut() {
                    t.observe(step.innovation, step.p_pred, kf.r);
                    // Retune every 20 samples from the running estimates.
                    if (i + 1) % 20 == 0 {
                        if let Some(r_hat) = t.estimate_r(1e-9) {
                            let r_new = (0.5 * kf.r + 0.5 * r_hat)
                                .clamp(0.1 * r0, 10.0 * r0);
                            let _ = kf.retune(kf.q, r_new);
                        }
                        if let Some(nis) = t.q_scale_factor() {
                            // Small damped step toward consistency:
                            // Q grows when NIS > 1, shrinks when NIS < 1.
                            let factor = (1.0 + 0.1 * (nis - 1.0)).clamp(0.9, 1.1);
                            let q_new = (kf.q * factor).clamp(0.01 * q0, 100.0 * q0);
                            let _ = kf.retune(q_new, kf.r);
                        }
                    }
                }
                steps.push(step);
            }
            let summary = summarise_scalar(&series, &steps);
            let out_path = o
                .output
                .clone()
                .unwrap_or_else(|| default_output_path(&o.input, "filtered.csv"));
            std::fs::write(&out_path, scalar_csv(&series, &steps))?;
            let summary_path = summary_path_for(&out_path);
            let title = format!(
                "kalman-filter-1d filter (scalar, q={}, r={}, p0={}) — {}",
                o.q,
                o.r,
                o.p0,
                o.input.display()
            );
            let rendered = summary.render(&title);
            std::fs::write(&summary_path, &rendered)?;
            out.push_str(&rendered);
            out.push_str(&format!("wrote {}\n", out_path.display()));
            out.push_str(&format!("wrote {}\n", summary_path.display()));
            if o.smooth {
                let smoothed = smooth_scalar(&steps);
                let smooth_path = default_output_path(&o.input, "smoothed.csv");
                std::fs::write(&smooth_path, scalar_smooth_csv(&series, &steps, &smoothed))?;
                let mut sm = summarise_scalar(&series, &steps);
                let est: Vec<f64> = smoothed.iter().map(|s| s.x).collect();
                if series.has_full_truth() {
                    let truth: Vec<f64> =
                        series.samples.iter().map(|s| s.truth.unwrap()).collect();
                    if let Ok(v) = rmse(&est, &truth) {
                        sm.rmse_smoothed = Some(v);
                    }
                }
                out.push_str(&format!("wrote {}\n", smooth_path.display()));
                if let Some(v) = sm.rmse_smoothed {
                    out.push_str(&format!("RMSE smoothed: {v:.6}\n"));
                }
            }
            if let Some(t) = &tracker {
                out.push_str(&format!(
                    "adaptive: final R estimate {:.6}, mean NIS {:.4}\n",
                    t.estimate_r(1e-9).unwrap_or(f64::NAN),
                    t.mean_nis
                ));
            }
        }
        "cv" => {
            let x0 = o.x0.unwrap_or(series.samples[0].z);
            let mut kf = CvKalman::new(x0, 0.0, o.p0, o.q, o.r)?;
            let dts = series.dts();
            let mut steps = Vec::with_capacity(series.len());
            for (i, s) in series.samples.iter().enumerate() {
                let dt = if i == 0 { 1.0 } else { dts[i].max(1e-9) };
                steps.push(kf.step(s.z, dt)?);
            }
            let summary = summarise_cv(&series, &steps);
            let out_path = o
                .output
                .clone()
                .unwrap_or_else(|| default_output_path(&o.input, "filtered.csv"));
            std::fs::write(&out_path, cv_csv(&series, &steps))?;
            let title = format!(
                "kalman-filter-1d filter (cv, q={}, r={}, p0={}) — {}",
                o.q,
                o.r,
                o.p0,
                o.input.display()
            );
            let rendered = summary.render(&title);
            let p = summary_path_for(&out_path);
            std::fs::write(&p, &rendered)?;
            out.push_str(&rendered);
            out.push_str(&format!("wrote {}\n", out_path.display()));
            out.push_str(&format!("wrote {}\n", p.display()));
            if o.smooth {
                let smoothed = smooth_cv(&steps, &dts_for_smooth(&series));
                let smooth_path = default_output_path(&o.input, "smoothed.csv");
                let mut text = String::from("t,z,truth,x,v,p,p_v,x_smooth,v_smooth,p_smooth\n");
                for ((s, st), sm) in series
                    .samples
                    .iter()
                    .zip(steps.iter())
                    .zip(smoothed.iter())
                {
                    match s.truth {
                        Some(truth) => text.push_str(&format!(
                            "{},{},{},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}\n",
                            s.t, s.z, truth, st.x[0], st.x[1], st.p[0][0], st.p[1][1],
                            sm.x[0], sm.x[1], sm.p[0][0]
                        )),
                        None => text.push_str(&format!(
                            "{},{},,{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}\n",
                            s.t, s.z, st.x[0], st.x[1], st.p[0][0], st.p[1][1],
                            sm.x[0], sm.x[1], sm.p[0][0]
                        )),
                    }
                }
                std::fs::write(&smooth_path, text)?;
                out.push_str(&format!("wrote {}\n", smooth_path.display()));
                if series.has_full_truth() {
                    let truth: Vec<f64> =
                        series.samples.iter().map(|s| s.truth.unwrap()).collect();
                    let est: Vec<f64> = smoothed.iter().map(|s| s.x[0]).collect();
                    if let Ok(v) = rmse(&est, &truth) {
                        out.push_str(&format!("RMSE smoothed: {v:.6}\n"));
                    }
                }
            }
        }
        other => return Err(Cli(format!("unknown filter {other:?}"))),
    }
    Ok(out)
}

/// `dts` for the smoother: the step *into* each sample, with the
/// first sample's dt replaced by the second sample's (the smoother
/// only uses `dts[k+1]`, so index 0 is never read).
fn dts_for_smooth(series: &TimeSeries) -> Vec<f64> {
    let mut dts = series.dts();
    if dts.len() > 1 {
        dts[0] = dts[1];
    } else if !dts.is_empty() {
        dts[0] = 1.0;
    }
    dts
}

fn run_generate(o: GenerateOptions) -> Result<String> {
    let scenario = parse_scenario(&o.scenario)?;
    let mut params = SynthParams::defaults(scenario);
    params.n = o.n;
    params.dt = o.dt;
    params.noise_std = o.noise_std;
    params.seed = o.seed;
    if let Some(v) = o.p1 {
        params.p1 = v;
    }
    if let Some(v) = o.p2 {
        params.p2 = v;
    }
    let series = generate(&params)?;
    let out_path = o
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{}.csv", o.scenario)));
    std::fs::write(&out_path, to_csv(&series))?;
    Ok(format!(
        "generated {} samples of scenario {:?} (seed {}) → {}\n",
        series.len(),
        scenario,
        params.seed,
        out_path.display()
    ))
}

fn run_compare(o: CompareOptions) -> Result<String> {
    let series = read_series(&o.input)?;
    if !series.has_full_truth() {
        return Err(KalmanError::InsufficientData(
            "compare needs ground truth in every row of the input CSV".into(),
        ));
    }
    let truth: Vec<f64> = series.samples.iter().map(|s| s.truth.unwrap()).collect();
    let raw = series.measurements();
    let raw_rmse = rmse(&raw, &truth)?;

    // Scalar Kalman filter.
    let kf = ScalarKalman::new(raw[0], o.p0, o.q, o.r)?;
    let scalar_steps = run_scalar_series(kf, &raw);
    let scalar_est: Vec<f64> = scalar_steps.iter().map(|s| s.x).collect();
    let scalar_rmse = rmse(&scalar_est, &truth)?;

    // Constant-velocity Kalman filter.
    let mut cv = CvKalman::new(raw[0], 0.0, o.p0, o.q, o.r)?;
    let dts = series.dts();
    let mut cv_est = Vec::with_capacity(series.len());
    for (i, s) in series.samples.iter().enumerate() {
        let dt = if i == 0 { 1.0 } else { dts[i].max(1e-9) };
        let step = cv.step(s.z, dt)?;
        cv_est.push(step.x[0]);
    }
    let cv_rmse = rmse(&cv_est, &truth)?;

    // Alpha-beta tracker.
    let mut ab = match (o.alpha, o.beta) {
        (Some(a), Some(b)) => AlphaBeta::new(raw[0], 0.0, a, b)?,
        _ => AlphaBeta::from_tracking_index(raw[0], 0.0, o.lambda.unwrap_or(1.0))?,
    };
    let mut ab_est = Vec::with_capacity(series.len());
    for (i, s) in series.samples.iter().enumerate() {
        let dt = if i == 0 { 1.0 } else { dts[i].max(1e-9) };
        let step = ab.step(s.z, dt)?;
        ab_est.push(step.x);
    }
    let ab_rmse = rmse(&ab_est, &truth)?;

    let mut out = String::new();
    out.push_str(&format!(
        "compare — {} ({} samples, q={}, r={}, p0={})\n",
        o.input.display(),
        series.len(),
        o.q,
        o.r,
        o.p0
    ));
    out.push_str(&format!("  raw measurements : RMSE {raw_rmse:.6}\n"));
    out.push_str(&format!(
        "  scalar Kalman    : RMSE {scalar_rmse:.6} ({:.2}x better than raw)\n",
        raw_rmse / scalar_rmse
    ));
    out.push_str(&format!(
        "  CV Kalman        : RMSE {cv_rmse:.6} ({:.2}x better than raw)\n",
        raw_rmse / cv_rmse
    ));
    out.push_str(&format!(
        "  alpha-beta (a={:.3}, b={:.3}) : RMSE {ab_rmse:.6} ({:.2}x better than raw)\n",
        ab.alpha, ab.beta,
        raw_rmse / ab_rmse
    ));
    let best = [("scalar Kalman", scalar_rmse), ("CV Kalman", cv_rmse), ("alpha-beta", ab_rmse)]
        .into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    out.push_str(&format!("best: {} (RMSE {:.6})\n", best.0, best.1));
    Ok(out)
}

fn run_batch_cmd(o: BatchOptions) -> Result<String> {
    let output_dir = o.output_dir.clone().unwrap_or_else(|| {
        o.input_dir.join("filtered")
    });
    let filter = if o.filter == "cv" {
        BatchFilter::Cv
    } else {
        BatchFilter::Scalar
    };
    let params = BatchParams {
        filter,
        q: o.q,
        r: o.r,
        p0: o.p0,
    };
    let results = run_batch(&o.input_dir, &output_dir, params)?;
    let ok = results.iter().filter(|r| r.outcome.is_ok()).count();
    let mut out = String::new();
    out.push_str(&render_batch_table(&results));
    out.push_str(&format!(
        "batch: {ok}/{} files filtered → {}\n",
        results.len(),
        output_dir.display()
    ));
    Ok(out)
}

/// The hand-checkable example, computed live by the real filter code
/// (so the printed numbers can never drift from the implementation).
fn handcheck_text() -> String {
    let mut kf = ScalarKalman::new(0.0, 1.0, 0.01, 0.1).unwrap();
    let mut out = String::from(
        "hand-check: scalar Kalman filter, x0=0, P0=1, Q=0.01, R=0.1\n\
         measurements z = [1.0, 1.1, 0.9]\n\n\
         step 1: predict  x⁻=0, P⁻=1.01; K=1.01/1.11=0.9099\n\
         \x20       update   x=0+0.9099·(1.0−0)=0.9099, P=(1−0.9099)·1.01=0.0910\n\
         step 2: predict  x⁻=0.9099, P⁻=0.1010; K=0.1010/0.2010=0.5025\n\
         \x20       update   x=0.9099+0.5025·(1.1−0.9099)=1.0054, P=0.0502\n\
         step 3: predict  x⁻=1.0054, P⁻=0.0602; K=0.0602/0.1602=0.3760\n\
         \x20       update   x=1.0054+0.3760·(0.9−1.0054)=0.9658, P=0.0376\n\n\
         live run (same numbers, computed by the filter itself):\n",
    );
    for z in [1.0, 1.1, 0.9] {
        let s = kf.step_plain(z);
        out.push_str(&format!(
            "  z={z:.1}: x_pred={:.4} P_pred={:.4} K={:.4} innovation={:+.4} → x={:.4} P={:.4}\n",
            s.x_pred, s.p_pred, s.gain, s.innovation, s.x, s.p
        ));
    }
    out.push_str(&format!(
        "\nsteady-state gain K∞ = {:.4}, P∞ = {:.4}\n",
        kf.steady_state_gain(),
        kf.steady_state_variance()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(|w| w.to_string()).collect()
    }

    #[test]
    fn parses_filter_with_equals_and_space_forms() {
        let cmd = parse_args(&args("prog filter in.csv --q=0.02 --r 0.3 --smooth")).unwrap();
        match cmd {
            Command::Filter(o) => {
                assert_eq!(o.q, 0.02);
                assert_eq!(o.r, 0.3);
                assert!(o.smooth);
            }
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn parses_generate_defaults_and_overrides() {
        let cmd = parse_args(&args("prog generate step --n 50 --seed 7")).unwrap();
        match cmd {
            Command::Generate(o) => {
                assert_eq!(o.scenario, "step");
                assert_eq!(o.n, 50);
                assert_eq!(o.seed, 7);
            }
            _ => panic!("expected Generate"),
        }
    }

    #[test]
    fn parses_adaptive_flag_with_and_without_lambda() {
        let cmd = parse_args(&args("prog filter in.csv --adaptive")).unwrap();
        match cmd {
            Command::Filter(o) => assert_eq!(o.adaptive, Some(0.98)),
            _ => panic!("expected Filter"),
        }
        let cmd = parse_args(&args("prog filter in.csv --adaptive 0.95")).unwrap();
        match cmd {
            Command::Filter(o) => assert_eq!(o.adaptive, Some(0.95)),
            _ => panic!("expected Filter"),
        }
    }

    #[test]
    fn rejects_unknown_subcommand_and_filter() {
        assert!(parse_args(&args("prog frobnicate")).is_err());
        assert!(parse_args(&args("prog filter in.csv --filter bogus")).is_err());
        assert!(parse_args(&args("prog filter")).is_err());
    }

    #[test]
    fn handcheck_output_contains_the_hand_computed_values() {
        let text = handcheck_text();
        assert!(text.contains("0.9099"));
        assert!(text.contains("1.0054"));
        assert!(text.contains("0.9658"));
        assert!(text.contains("K∞ = 0.2702"));
    }
}
