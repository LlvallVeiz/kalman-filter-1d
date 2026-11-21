# Research notes — Kalman filtering in one dimension

This document records the theory this crate implements, why each
design choice was made, and where every claim comes from. All
references below are real publications; the two DOI lookups that
could have been misremembered (Rauch–Tung–Striebel, Mehra) were
verified against search results on 2026-11-21 before writing.

## 1. The problem

A sensor reports a quantity `z_k` that is the true state `x_k` plus
noise. The state itself evolves in time. The Kalman filter (Kalman,
1960) is the recursive, minimum-mean-square-error estimator for this
problem when the dynamics are linear and all noise is Gaussian: it
keeps a state estimate and its error covariance, and updates both
with each new measurement. It is optimal in that setting, and it is
the workhorse of navigation, tracking, sensor fusion and signal
smoothing (Welch & Bishop, 2006; Bar-Shalom, Li & Kirubarajan, 2001).

## 2. The scalar filter (random walk)

Model (Welch & Bishop, 2006, Tables 1-1/1-2, reduced to one state):

```text
x_k = x_{k-1} + w_{k-1},  w ~ N(0, Q)    process
z_k = x_k + v_k,          v ~ N(0, R)    measurement
```

Time update: `x⁻ = x`, `P⁻ = P + Q`. Measurement update:

```text
K = P⁻ / (P⁻ + R)
x = x⁻ + K·(z − x⁻)
P = (1 − K)·P⁻
```

The gain is the whole story: `R ≪ P⁻` → `K ≈ 1` (trust the
measurement); `R ≫ P⁻` → `K ≈ 0` (trust the prediction). The
steady-state gain has a closed form: with `S = P⁻` at steady state,
`S² − Q·S − Q·R = 0`, so `S = (Q + √(Q² + 4QR))/2` and
`K∞ = S/(S + R)`. The crate asserts the closed form against 500
iterated steps (`src/scalar.rs`).

**When it works:** signals that wander slowly with no systematic
trend — a room-temperature sensor, a slowly drifting bias, a
quasi-static level. **When it fails:** any signal with a trend. On
the ramp example the scalar filter lags by a constant amount and its
RMSE (0.4998) is no better than the raw measurements (0.4969), with
NEES 5.50 — the filter's own covariance says the error should be
small while the lag makes it large. That failure is the motivation
for the constant-velocity model.

## 3. The constant-velocity (CV) filter

State `[x, v]`, `F = [[1, dt], [0, 1]]`, `H = [1, 0]`
(Bar-Shalom, Li & Kirubarajan, 2001, §6.2; Gelb, 1974). The process
noise is the discrete white-noise acceleration (DWNA) model: the
acceleration is constant over one sampling interval,
`a ~ N(0, q)`, giving

```text
Q = q · [[dt⁴/4, dt³/2],
         [dt³/2, dt²  ]]
```

The crate also offers the discrete alternative
`Q = q·[[dt², dt],[dt, 1]]` (per-step velocity-increment variance),
selectable via `CvKalman::with_discrete_q`. The two forms differ only
in how `q` is interpreted; both are documented in `src/cv.rs`. The 2×2 arithmetic is written out by hand (no linear
algebra crate), so every line can be checked against the textbook
equations — and the first step is hand-checked in
`examples/hand_check.md`.

On the ramp example the CV filter's RMSE is 0.2583 (1.92× better
than raw) where the scalar filter manages 0.4998: the velocity
state carries the trend.

## 4. The α–β filter (baseline)

Benedict & Bordner (1962) proposed the fixed-gain tracker:

```text
x⁻ = x + v·dt;  r = z − x⁻
x  = x⁻ + α·r;  v = v + (β/dt)·r
```

Same two states as the CV filter, same measurements, but constant
gains and no covariance — the standard baseline a Kalman filter must
beat. Kalata (1984) related the gains to a *tracking index*
`λ = σ_w·dt²/σ_v` (process-noise scale over measurement-noise scale)
through `β(α) = 2(2−α) − 4√(1−α)` and `λ² = β²/(α(2−α))`, solved
here by bisection (`src/alpha_beta.rs`). `λ < 2` is required: the
steady-state solution saturates at `α = 1, β = 2`.

On the step example (`λ = 1`, `α = 0.913`, `β = 0.996`) the α–β
tracker scores RMSE 0.4632 — better than raw (0.4969) but worse than
the scalar Kalman filter (0.3371). On the ramp it scores 0.4609
against the CV filter's 0.3088. The Kalman filters win because their
gain adapts: large at the start (uncertain prior), small at steady
state (confident estimate).

## 5. The Rauch–Tung–Striebel smoother

A filter uses only past measurements; a smoother re-processes the
whole batch backwards and uses future measurements too, so it is
strictly more accurate at every interior point (Rauch, Tung &
Striebel, 1965):

```text
G_k   = P_k · Fᵀ / P⁻_{k+1}
x̂ˢ_k = x̂_k + G_k·(x̂ˢ_{k+1} − x̂⁻_{k+1})
Pˢ_k  = P_k + G_k²·(Pˢ_{k+1} − P⁻_{k+1})
```

Measured on the examples: step 0.3371 → 0.2963, sine 0.1652 →
0.0971, ramp (CV) 0.2583 → 0.1500, temperature 0.1006 → 0.0626,
altitude (CV) 0.5351 → 0.2956. The smoother's variance never exceeds
the filter's (asserted in tests). Cost: it needs the complete forward
pass first, so it cannot run online.

## 6. Adaptive noise estimation

The filter is only optimal when `Q` and `R` match reality. The crate
implements innovation-based covariance matching (Mehra, 1970): a
correctly tuned filter has white innovations with variance
`S_k = P⁻_k + R`, so

```text
R̂ = mean(ν²) − mean(P⁻)
```

with an exponentially weighted mean (forgetting factor λ) so the
estimate can follow drifting noise. `Q` is corrected from the mean
normalised innovation squared (NIS): NIS > 1 means the state moves
faster than the model allows, so `Q` grows.

**A real failure found while building this:** the first version of
the CLI's `--adaptive` mode applied the raw estimates in full every
20 steps. It diverged: `Q` grew, which inflated `mean(P⁻)`, which
drove `R̂` negative (floored at 1e-9), which collapsed the gain to 1
and the covariance to ~0 — NEES 27,400,883 on the temperature
example. The shipped version damps both updates (half old, half new)
and clamps them to a decade around the starting values
(`src/cli.rs`). With that fix, adaptive mode on the temperature
sensor (starting from `Q = 0.01, R = 0.04`) gives RMSE 0.1133 and
NEES 1.07 (consistent), against the fixed-parameter run's RMSE
0.1046 with NEES 0.70 (inconsistent): adaptation trades a small
RMSE cost for an honest covariance. The estimates are reported, not
hidden, so the user can see whether adaptation converged
(`adaptive: final R estimate 0.021973, mean NIS 0.9541` — true
`R = 0.04`, same order of magnitude).

## 7. Validation: RMSE and NEES

Two questions (Bar-Shalom, Li & Kirubarajan, 2001, §5.4):

1. **Accuracy** — RMSE of the estimates vs ground truth, compared
   with the RMSE of the raw measurements. A filter that does not
   beat the raw data is useless.
2. **Consistency** — the normalised estimation error squared (NEES).
   For a filter with `n` states, each NEES sample is chi-square with
   `n` degrees of freedom *if* the covariance is honest, so
   `N·mean(NEES)` is chi-square with `N·n` dof. The 95% interval for
   the mean is `[χ²_{0.025,N·n}/N, χ²_{0.975,N·n}/N]` — centred on
   `n`, not on 1 (a 2-dof filter's mean NEES should be ≈ 2). This
   crate caught its own normalisation bug here during development:
   the first version divided by `N·n` and reported a *correct* CV
   filter (mean NEES 1.98) as inconsistent; the fix is asserted by
   the unit test `nees_report_two_dof_interval_is_centred_on_two`.

Chi-square quantiles are computed exactly by bisection on the
regularised incomplete gamma function (Numerical Recipes' series /
continued-fraction expansions), verified against table values:
`χ²_0.975(1) = 5.02389`, `χ²_0.025(1) = 0.000982`,
`χ²_0.975(2) = 7.37776`, `χ²_0.025(2) = 0.05064`,
`χ²_0.975(100) = 129.561`, `χ²_0.025(100) = 74.222`
(cross-checked with the Poisson/gamma identity and the
Wilson–Hilferty approximation).

**A caveat the examples exposed:** the NEES test assumes
independent samples, but consecutive NEES values of a filter are
correlated (the filter has memory). On a 200-sample run the
chi-square interval is too narrow: a Monte-Carlo check (50 seeds)
of the matched altitude filter gave a mean NEES of 1.01 on average,
but only 56% of individual 200-sample runs fell inside the 95%
interval. With 1000 samples the seed-42 run lands at 1.048
(inside). That is why the altitude example uses `n = 1000` while
the others use 200–288 samples. Treat a single short-run NEES
verdict as a guide, not a proof — and prefer long runs.

## 8. Random numbers

No `rand` crate: the project builds fully offline. The generator is
SplitMix64 (Stafford, 2016) with a Box–Muller transform (Box &
Muller, 1958) for Gaussians. Every synthetic dataset is reproducible
from its integer seed (verified by the determinism tests).

## References

1. Kalman, R. E. (1960). "A New Approach to Linear Filtering and
   Prediction Problems." *Transactions of the ASME — Journal of
   Basic Engineering*, 82(Series D), 35–45.
   DOI: [10.1115/1.3662552](https://doi.org/10.1115/1.3662552) ·
   PDF: <https://www.cs.cmu.edu/~./motionplanning/papers/sbp_papers/k/Kalman1960.pdf>
2. Welch, G., & Bishop, G. (2006). *An Introduction to the Kalman
   Filter*. UNC-Chapel Hill, TR 95-041 (updated July 24, 2006).
   <https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/motionplanning/papers/sbp_papers/kalman/welch_intro_kalman.pdf>
3. Rauch, H. E., Tung, F., & Striebel, C. T. (1965). "Maximum
   likelihood estimates of linear dynamic systems." *AIAA Journal*,
   3(8), 1445–1450. DOI: [10.2514/3.3166](https://doi.org/10.2514/3.3166)
4. Mehra, R. K. (1970). "On the identification of variances and
   adaptive Kalman filtering." *IEEE Transactions on Automatic
   Control*, 15(2), 175–184.
   DOI: [10.1109/TAC.1970.1099422](https://doi.org/10.1109/TAC.1970.1099422)
5. Benedict, T. R., & Bordner, G. W. (1962). "Synthesis of an
   optimal set of radar track-while-scan smoothing equations."
   *IRE Transactions on Automatic Control*, 7(4), 27–32.
6. Kalata, P. R. (1984). "The tracking index: a generalized
   parameter for α–β and α–β–γ target trackers." *IEEE
   Transactions on Aerospace and Electronic Systems*, 20(2), 174–182.
7. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
   *Estimation with Applications to Tracking and Navigation*.
   Wiley. (Ch. 5: consistency testing; Ch. 6: CV/DWNA models.)
8. Gelb, A. (Ed.) (1974). *Applied Optimal Estimation*. MIT Press.
9. Simon, D. (2006). *Optimal State Estimation*. Wiley.
10. Box, G. E. P., & Muller, M. E. (1958). "A note on the
    generation of random normal deviates." *The Annals of
    Mathematical Statistics*, 29(2), 610–611.
11. Wilson, E. B., & Hilferty, M. M. (1931). "The distribution of
    chi-square." *Proceedings of the National Academy of Sciences*,
    17(12), 684–688. (Approximation used to cross-check the
    chi-square quantiles.)
12. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery,
    B. P. (2007). *Numerical Recipes* (3rd ed.). Cambridge
    University Press. (Incomplete gamma `gammp`/`gcf`, Lanczos
    `gammln` — the implementations in `src/metrics.rs`.)
13. Stafford, C. (2016). SplitMix64 generator (constants as in the
    original `splitmix64` C code); see also Wellons, C.,
    "SplitMix64", <https://nullprogram.com/blog/2018/07/31/>.
