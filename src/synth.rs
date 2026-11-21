//! Synthetic data generation — reproducible test signals.
//!
//! Every scenario produces a [`TimeSeries`] with known ground truth,
//! so filter accuracy (RMSE) and consistency (NEES) can be measured
//! instead of assumed. All randomness comes from [`GaussianRng`] with
//! an explicit integer seed: the same seed reproduces the same file
//! bit-for-bit on any machine.
//!
//! Scenarios:
//!
//! * [`Scenario::Step`] — a constant that jumps once (tests tracking
//!   of abrupt changes; the filter must lag, the smoother must not).
//! * [`Scenario::Sine`] — a sinusoid (tests tracking of a smooth,
//!   continuously moving signal).
//! * [`Scenario::Ramp`] — a linear trend (the scalar filter lags by a
//!   constant amount; the CV filter should not).
//! * [`Scenario::Temperature`] — a realistic room-temperature sensor:
//!   slow diurnal swing + linear drift + measurement noise. The
//!   scenario the scalar filter was built for.
//! * [`Scenario::Altitude`] — a random-walk altitude (barometric
//!   sensor on a slowly drifting platform) with a velocity component,
//!   for the CV filter.

use crate::errors::{require_non_negative, require_positive, Result};
use crate::rng::GaussianRng;
use crate::series::{Sample, TimeSeries};

/// Which synthetic signal to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// Constant `before`, jumps to `after` at `step_at` seconds.
    Step,
    /// `offset + amplitude·sin(2π·freq·t)`.
    Sine,
    /// `intercept + slope·t`.
    Ramp,
    /// Room-temperature sensor: diurnal sine + linear drift.
    Temperature,
    /// Random-walk altitude (position + velocity, for the CV filter).
    Altitude,
}

/// Parameters shared by all scenarios.
#[derive(Debug, Clone, Copy)]
pub struct SynthParams {
    /// Scenario to generate.
    pub scenario: Scenario,
    /// Number of samples.
    pub n: usize,
    /// Sampling interval in seconds.
    pub dt: f64,
    /// Standard deviation of the measurement noise.
    pub noise_std: f64,
    /// Integer seed for the Gaussian generator.
    pub seed: u64,
    /// Scenario-specific parameter 1 (see [`Scenario`] docs):
    /// step size / sine amplitude / ramp slope / drift rate / accel std.
    pub p1: f64,
    /// Scenario-specific parameter 2: step time / sine frequency /
    /// unused / diurnal amplitude / initial velocity.
    pub p2: f64,
}

impl SynthParams {
    /// Sensible defaults per scenario; callers override the fields
    /// they care about.
    pub fn defaults(scenario: Scenario) -> Self {
        match scenario {
            Scenario::Step => Self {
                scenario,
                n: 200,
                dt: 1.0,
                noise_std: 0.5,
                seed: 42,
                p1: 5.0,  // jump size
                p2: 100.0, // jump at t = 100 s
            },
            Scenario::Sine => Self {
                scenario,
                n: 200,
                dt: 0.1,
                noise_std: 0.3,
                seed: 42,
                p1: 2.0,  // amplitude
                p2: 0.05, // frequency (Hz) → 20 s period
            },
            Scenario::Ramp => Self {
                scenario,
                n: 200,
                dt: 1.0,
                noise_std: 0.5,
                seed: 42,
                p1: 0.1, // slope per second
                p2: 0.0, // intercept
            },
            Scenario::Temperature => Self {
                scenario,
                n: 288,
                dt: 300.0, // 5-minute samples, 24 h total
                noise_std: 0.2,
                seed: 42,
                p1: 0.02, // drift °C per hour
                p2: 1.5,  // diurnal amplitude °C
            },
            Scenario::Altitude => Self {
                scenario,
                n: 200,
                dt: 1.0,
                noise_std: 1.0,
                seed: 42,
                p1: 0.05, // acceleration std (m/s²)
                p2: 0.0,  // initial velocity (m/s)
            },
        }
    }

    /// Validate the parameters before generation.
    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            return Err(crate::errors::KalmanError::InvalidParameter(
                "n must be > 0".to_string(),
            ));
        }
        require_positive(self.dt, "dt")?;
        require_non_negative(self.noise_std, "noise_std")?;
        Ok(())
    }
}

/// Ground truth of a scenario at time `t` (and step index `k`).
fn truth_at(params: &SynthParams, t: f64, k: usize) -> f64 {
    match params.scenario {
        Scenario::Step => {
            if t < params.p2 {
                0.0
            } else {
                params.p1
            }
        }
        Scenario::Sine => params.p1 * (2.0 * std::f64::consts::PI * params.p2 * t).sin(),
        Scenario::Ramp => params.p2 + params.p1 * t,
        Scenario::Temperature => {
            // 25 °C base, diurnal swing with a 24 h period, plus drift.
            25.0 + params.p2 * (2.0 * std::f64::consts::PI * t / 86_400.0).sin()
                + params.p1 * (t / 3600.0)
        }
        Scenario::Altitude => {
            // Random walk is stateful; its truth is produced in
            // `generate` and this arm is never reached there. The
            // `k`-indexed fallback keeps the function total.
            let _ = k;
            100.0
        }
    }
}

/// Generate a synthetic series. The returned series always carries
/// ground truth.
pub fn generate(params: &SynthParams) -> Result<TimeSeries> {
    params.validate()?;
    let mut rng = GaussianRng::new(params.seed);
    let mut samples = Vec::with_capacity(params.n);

    if params.scenario == Scenario::Altitude {
        // Stateful random walk in the discrete white-noise acceleration
        // (DWNA) form — the exact model the CV filter implements:
        //   x += v·dt + ½·a·dt²,  v += a·dt,  a ~ N(0, p1²)
        // so a CV filter with q = p1² is matched to this truth.
        let mut x = 100.0f64;
        let mut v = params.p2;
        for k in 0..params.n {
            let t = k as f64 * params.dt;
            if k > 0 {
                let a = rng.gaussian(0.0, params.p1);
                x += v * params.dt + 0.5 * a * params.dt * params.dt;
                v += a * params.dt;
            }
            let z = x + rng.gaussian(0.0, params.noise_std);
            samples.push(Sample { t, z, truth: Some(x) });
        }
        return TimeSeries::new(samples);
    }

    for k in 0..params.n {
        let t = k as f64 * params.dt;
        let truth = truth_at(params, t, k);
        let z = truth + rng.gaussian(0.0, params.noise_std);
        samples.push(Sample { t, z, truth: Some(truth) });
    }
    TimeSeries::new(samples)
}

/// Parse a scenario name as used on the command line.
pub fn parse_scenario(name: &str) -> Result<Scenario> {
    match name.to_ascii_lowercase().as_str() {
        "step" => Ok(Scenario::Step),
        "sine" => Ok(Scenario::Sine),
        "ramp" => Ok(Scenario::Ramp),
        "temperature" | "temp" => Ok(Scenario::Temperature),
        "altitude" => Ok(Scenario::Altitude),
        other => Err(crate::errors::KalmanError::InvalidParameter(format!(
            "unknown scenario {other:?}; expected step, sine, ramp, temperature or altitude"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_for_a_seed() {
        let p = SynthParams::defaults(Scenario::Sine);
        let a = generate(&p).unwrap();
        let b = generate(&p).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_noise() {
        let mut p = SynthParams::defaults(Scenario::Step);
        let a = generate(&p).unwrap();
        p.seed = 1234;
        let b = generate(&p).unwrap();
        assert_ne!(a.samples[0].z, b.samples[0].z);
        // ...but the same truth.
        assert_eq!(a.samples[0].truth, b.samples[0].truth);
    }

    #[test]
    fn step_truth_jumps_at_the_right_time() {
        let p = SynthParams::defaults(Scenario::Step);
        let ts = generate(&p).unwrap();
        assert_eq!(ts.samples[99].truth, Some(0.0));
        assert_eq!(ts.samples[100].truth, Some(5.0));
        assert_eq!(ts.samples[199].truth, Some(5.0));
    }

    #[test]
    fn sine_truth_is_a_sine() {
        // Amplitude 2, frequency 0.05 Hz, dt 0.1: t = 5 s is a quarter
        // period → sin(π/2) = 1 → truth 2.0.
        let p = SynthParams::defaults(Scenario::Sine);
        let ts = generate(&p).unwrap();
        assert!((ts.samples[50].truth.unwrap() - 2.0).abs() < 1e-9);
        assert!((ts.samples[0].truth.unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn ramp_truth_is_linear() {
        let p = SynthParams::defaults(Scenario::Ramp);
        let ts = generate(&p).unwrap();
        // slope 0.1/s, dt 1 s: truth at k=10 is 1.0.
        assert!((ts.samples[10].truth.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn temperature_starts_at_25_and_drifts_up() {
        let p = SynthParams::defaults(Scenario::Temperature);
        let ts = generate(&p).unwrap();
        // t=0: 25 + 0 + 0 = 25. After 24 h the drift adds 0.48 °C and
        // the diurnal sine returns to 0.
        assert!((ts.samples[0].truth.unwrap() - 25.0).abs() < 1e-9);
        let last = ts.samples.last().unwrap().truth.unwrap();
        assert!((last - 25.48).abs() < 0.05, "last truth {last}");
    }

    #[test]
    fn altitude_truth_is_a_random_walk() {
        let p = SynthParams::defaults(Scenario::Altitude);
        let ts = generate(&p).unwrap();
        assert_eq!(ts.len(), 200);
        assert!((ts.samples[0].truth.unwrap() - 100.0).abs() < 1e-12);
        // The walk must actually move over 200 steps.
        let last = ts.samples.last().unwrap().truth.unwrap();
        assert!((last - 100.0).abs() > 0.1, "walk did not move: {last}");
    }

    #[test]
    fn zero_noise_measurements_equal_truth() {
        let mut p = SynthParams::defaults(Scenario::Sine);
        p.noise_std = 0.0;
        let ts = generate(&p).unwrap();
        for s in &ts.samples {
            assert_eq!(s.z, s.truth.unwrap());
        }
    }

    #[test]
    fn parse_scenario_names() {
        assert_eq!(parse_scenario("step").unwrap(), Scenario::Step);
        assert_eq!(parse_scenario("TEMP").unwrap(), Scenario::Temperature);
        assert_eq!(parse_scenario("temp").unwrap(), Scenario::Temperature);
        assert!(parse_scenario("nope").is_err());
    }

    #[test]
    fn validate_rejects_bad_params() {
        let mut p = SynthParams::defaults(Scenario::Step);
        p.n = 0;
        assert!(generate(&p).is_err());
        let mut p = SynthParams::defaults(Scenario::Step);
        p.dt = 0.0;
        assert!(generate(&p).is_err());
        let mut p = SynthParams::defaults(Scenario::Step);
        p.noise_std = -1.0;
        assert!(generate(&p).is_err());
    }
}
