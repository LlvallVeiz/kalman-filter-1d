//! Error type shared by every module of `kalman-filter-1d`.
//!
//! The crate is std-only, so errors are plain values: no `anyhow`, no
//! `thiserror`. Every fallible public function returns
//! `Result<T, KalmanError>` and every error message names the offending
//! input so a CLI user can fix it without reading the source.

use std::fmt;

/// All errors this crate can produce.
#[derive(Debug, Clone, PartialEq)]
pub enum KalmanError {
    /// A numeric parameter is outside its valid range
    /// (e.g. negative variance, non-positive `dt`).
    InvalidParameter(String),
    /// CSV text could not be parsed (bad header, bad number, wrong columns).
    CsvParse(String),
    /// A file could not be read or written.
    Io(String),
    /// The command line could not be understood.
    Cli(String),
    /// The requested operation needs data that is not there
    /// (empty series, missing ground truth, too few samples).
    InsufficientData(String),
}

impl fmt::Display for KalmanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KalmanError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            KalmanError::CsvParse(msg) => write!(f, "CSV parse error: {msg}"),
            KalmanError::Io(msg) => write!(f, "I/O error: {msg}"),
            KalmanError::Cli(msg) => write!(f, "CLI error: {msg}"),
            KalmanError::InsufficientData(msg) => write!(f, "insufficient data: {msg}"),
        }
    }
}

impl std::error::Error for KalmanError {}

impl From<std::io::Error> for KalmanError {
    fn from(err: std::io::Error) -> Self {
        KalmanError::Io(err.to_string())
    }
}

/// Convenience alias used by every public function in the crate.
pub type Result<T> = std::result::Result<T, KalmanError>;

/// Helper: require a finite, strictly positive variance-like parameter.
pub fn require_positive(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(KalmanError::InvalidParameter(format!(
            "{name} must be finite and > 0, got {value}"
        )));
    }
    Ok(())
}

/// Helper: require a finite, non-negative parameter.
pub fn require_non_negative(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(KalmanError::InvalidParameter(format!(
            "{name} must be finite and >= 0, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_problem() {
        let e = KalmanError::InvalidParameter("q must be finite and > 0, got -1".into());
        assert!(e.to_string().contains("q must be finite"));
    }

    #[test]
    fn require_positive_rejects_zero_negative_and_nan() {
        assert!(require_positive(1.0, "r").is_ok());
        assert!(require_positive(0.0, "r").is_err());
        assert!(require_positive(-2.0, "r").is_err());
        assert!(require_positive(f64::NAN, "r").is_err());
        assert!(require_positive(f64::INFINITY, "r").is_err());
    }

    #[test]
    fn io_error_converts() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let e = KalmanError::from(io);
        assert!(matches!(e, KalmanError::Io(_)));
    }
}
