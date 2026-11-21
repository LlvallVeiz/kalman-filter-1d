//! Seeded pseudo-random number generation (std-only).
//!
//! The crate deliberately has no `rand` dependency so it builds fully
//! offline. The generator is SplitMix64 (Stafford, 2016) — a fast,
//! well-tested 64-bit generator with a 2^64 period — combined with the
//! Box–Muller transform (Box & Muller, 1958) for Gaussian samples.
//! Every synthetic dataset in this project is reproducible from its
//! integer seed: the same seed always produces the same numbers, on any
//! platform, because the arithmetic is exact integer math plus one
//! `f64` transform.

/// SplitMix64 generator.
///
/// Reference: Chris Wellons, "SplitMix64",
/// <https://nullprogram.com/blog/2018/07/31/> (analysis of the
/// generator published by Sebastiano Vigna); the constants below are
/// the canonical ones from Stafford's original `splitmix64` C code.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from any 64-bit seed. Seed 0 is legal and is
    /// mixed by the first output step, so it does not get stuck.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        // Golden-gamma increment; the mixing function is a fixed
        // bijection on u64, so the sequence has full period 2^64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)` with 53 bits of precision
    /// (the full `f64` mantissa).
    pub fn next_f64(&mut self) -> f64 {
        // Take the top 53 bits, as `rand` does for `StandardUniform`.
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Uniform sample in `[low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

/// Gaussian (normal) sampler built on [`SplitMix64`] via Box–Muller.
///
/// The second variate of each Box–Muller pair is cached, so consecutive
/// calls alternate between a fresh transform and the cached spare —
/// exactly the standard implementation.
#[derive(Debug, Clone)]
pub struct GaussianRng {
    uniform: SplitMix64,
    spare: Option<f64>,
}

impl GaussianRng {
    /// Create a Gaussian sampler from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self {
            uniform: SplitMix64::new(seed),
            spare: None,
        }
    }

    /// One standard-normal sample, N(0, 1).
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        // Box–Muller: u1 in (0, 1] (never 0, so ln is safe), u2 in [0, 1).
        let mut u1 = self.uniform.next_f64();
        while u1 == 0.0 {
            u1 = self.uniform.next_f64();
        }
        let u2 = self.uniform.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    /// One Gaussian sample with the given mean and standard deviation.
    pub fn gaussian(&mut self, mean: f64, std: f64) -> f64 {
        debug_assert!(std >= 0.0, "gaussian std must be non-negative");
        mean + std * self.standard_normal()
    }

    /// Uniform sample in `[0, 1)` (pass-through to the inner generator).
    pub fn uniform(&mut self) -> f64 {
        self.uniform.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic_for_a_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix_different_seeds_differ() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniform_stays_in_unit_interval() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "uniform out of range: {u}");
        }
    }

    #[test]
    fn gaussian_is_deterministic_for_a_seed() {
        let mut a = GaussianRng::new(1234);
        let mut b = GaussianRng::new(1234);
        for _ in 0..1000 {
            assert_eq!(a.standard_normal(), b.standard_normal());
        }
    }

    #[test]
    fn gaussian_sample_moments_are_close_to_theory() {
        // 200k samples: mean within 0.01 and std within 0.01 of N(0,1)
        // with overwhelming probability; the tolerance is far above the
        // Monte-Carlo standard error (~0.0022), so this is a stable test.
        let mut rng = GaussianRng::new(20261121);
        let n = 200_000usize;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let v = rng.standard_normal();
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "variance {var}");
    }

    #[test]
    fn gaussian_with_mean_and_std() {
        let mut rng = GaussianRng::new(99);
        let n = 100_000usize;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += rng.gaussian(10.0, 2.0);
        }
        let mean = sum / n as f64;
        assert!((mean - 10.0).abs() < 0.05, "mean {mean}");
    }
}
