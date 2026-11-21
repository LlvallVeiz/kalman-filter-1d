# Hand-check: the scalar Kalman filter, worked by hand

This file proves the filter's arithmetic with numbers you can verify
on paper. The setup is the smallest non-trivial case:

- initial estimate `x₀ = 0`, initial variance `P₀ = 1`
- process noise `Q = 0.01`, measurement noise `R = 0.1`
- measurements `z = [1.0, 1.1, 0.9]`

## Step 1 — `z₁ = 1.0`

Predict (random walk, no control input):

```text
x⁻ = x = 0
P⁻ = P + Q = 1 + 0.01 = 1.01
```

Gain and update:

```text
K  = P⁻ / (P⁻ + R) = 1.01 / 1.11 = 0.9099
ν  = z − x⁻ = 1.0 − 0 = 1.0
x  = x⁻ + K·ν = 0 + 0.9099 · 1.0 = 0.9099
P  = (1 − K)·P⁻ = 0.0901 · 1.01 = 0.0910
```

## Step 2 — `z₂ = 1.1`

```text
x⁻ = 0.9099,  P⁻ = 0.0910 + 0.01 = 0.1010
K  = 0.1010 / 0.2010 = 0.5025
ν  = 1.1 − 0.9099 = 0.1901
x  = 0.9099 + 0.5025 · 0.1901 = 1.0054
P  = (1 − 0.5025) · 0.1010 = 0.0502
```

## Step 3 — `z₃ = 0.9`

```text
x⁻ = 1.0054,  P⁻ = 0.0502 + 0.01 = 0.0602
K  = 0.0602 / 0.1602 = 0.3760
ν  = 0.9 − 1.0054 = −0.1054
x  = 1.0054 + 0.3760 · (−0.1054) = 0.9658
P  = (1 − 0.3760) · 0.0602 = 0.0376
```

## Live check

The same three steps, computed by the filter itself
(`kalman-filter-1d handcheck`, output saved in `hand_check.txt`):

```text
hand-check: scalar Kalman filter, x0=0, P0=1, Q=0.01, R=0.1
measurements z = [1.0, 1.1, 0.9]

step 1: predict  x⁻=0, P⁻=1.01; K=1.01/1.11=0.9099
         update   x=0+0.9099·(1.0−0)=0.9099, P=(1−0.9099)·1.01=0.0910
step 2: predict  x⁻=0.9099, P⁻=0.1010; K=0.1010/0.2010=0.5025
         update   x=0.9099+0.5025·(1.1−0.9099)=1.0054, P=0.0502
step 3: predict  x⁻=1.0054, P⁻=0.0602; K=0.0602/0.1602=0.3760
         update   x=1.0054+0.3760·(0.9−1.0054)=0.9658, P=0.0376

live run (same numbers, computed by the filter itself):
  z=1.0: x_pred=0.0000 P_pred=1.0100 K=0.9099 innovation=+1.0000 → x=0.9099 P=0.0910
  z=1.1: x_pred=0.9099 P_pred=0.1010 K=0.5025 innovation=+0.1901 → x=1.0054 P=0.0502
  z=0.9: x_pred=1.0054 P_pred=0.0602 K=0.3760 innovation=-0.1054 → x=0.9658 P=0.0376

steady-state gain K∞ = 0.2702, P∞ = 0.0270
```

The live run matches the hand calculation to 4 decimals on every
quantity. The underlying values are asserted exactly in
`src/scalar.rs`, test `first_three_steps_match_hand_calculation`.

## Steady state

For this model the gain converges to a fixed value. Solving the
scalar Riccati equation `S² − Q·S − Q·R = 0` with `S = P⁻`:

```text
S  = (Q + √(Q² + 4QR)) / 2 = (0.01 + √(0.0001 + 0.004)) / 2 = 0.0370
K∞ = S / (S + R) = 0.0370 / 0.1370 = 0.2702
P∞ = (1 − K∞)·S = 0.0270
```

Verified independently in Python and asserted in
`src/scalar.rs`, test `steady_state_matches_iteration`
(500 iterated steps agree with the closed form to 1e-9).

## Constant-velocity filter — one hand-checked step

Setup: `x = [0, 0]`, `P = I`, `dt = 1`, `q = 0.01` (discrete
white-noise acceleration), `R = 0.1`, `z = 1.0`.

```text
Q = 0.01 · [[dt⁴/4, dt³/2], [dt³/2, dt²]] = [[0.0025, 0.005], [0.005, 0.01]]
P⁻ = F·P·Fᵀ + Q = [[2, 1], [1, 1]] + Q = [[2.0025, 1.005], [1.005, 1.01]]
S  = P⁻₀₀ + R = 2.1025
K  = [P⁻₀₀, P⁻₁₀] / S = [0.9524, 0.4780]
x  = [0, 0] + K · (1.0 − 0) = [0.9524, 0.4780]
P  = (I − K·H)·P⁻ = [[0.0952, 0.0478], [0.0478, 0.5296]]
```

Asserted in `src/cv.rs`, test `first_step_matches_hand_calculation`.

## α–β filter — one hand-checked step

Setup: `x = 0`, `v = 0`, `α = 0.5`, `β = 0.1`, `dt = 1`, `z = 1.0`:

```text
x⁻ = x + v·dt = 0
r  = z − x⁻ = 1.0
x  = x⁻ + α·r = 0.5
v  = v + (β/dt)·r = 0.1
```

Asserted in `src/alpha_beta.rs`, test `first_step_matches_hand_calculation`.
