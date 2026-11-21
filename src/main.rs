//! Binary entry point: parse the command line, run it, print the result.
//!
//! Exit code 0 on success, 1 on error (message on stderr).

use kalman_filter_1d::cli::{parse_args, run_command};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match parse_args(&args).and_then(run_command) {
        Ok(text) => {
            print!("{text}");
            ExitCode::from(0)
        }
        Err(e) => {
            eprintln!("kalman-filter-1d: error: {e}");
            eprintln!("Run `kalman-filter-1d --help` for usage.");
            ExitCode::from(1)
        }
    }
}
