#!/usr/bin/env bash
# Regenerate every file in examples/ from scratch with the release binary.
# Run from the project root:  bash examples/run_examples.sh
# All numbers in README.md come from this script's output — do not edit
# the generated files by hand.
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --quiet
B="../target/release/kalman-filter-1d"

# --- 1. Synthetic inputs (seed 42 everywhere, so every run reproduces) ---
"$B" generate step        --seed 42 --output step.csv
"$B" generate sine        --seed 42 --output sine.csv
"$B" generate ramp        --seed 42 --output ramp.csv
"$B" generate temperature --seed 42 --output temp_sensor.csv
# Altitude needs a longer run for a meaningful NEES verdict: with only
# 200 samples the NEES values are strongly correlated (filter memory)
# and the chi-square interval is too narrow — see README "Limitations".
"$B" generate altitude    --seed 42 --n 1000 --output altitude.csv

# --- 2. Filtered outputs (tuned Q/R per scenario; see README for why) ---
"$B" filter step.csv        --q 0.1    --r 0.25 --smooth
"$B" filter sine.csv        --q 0.01   --r 0.09 --smooth
"$B" filter ramp.csv        --filter cv --q 0.001 --r 0.25 --smooth
"$B" filter ramp.csv        --q 0.01   --r 0.25 --output ramp_scalar.filtered.csv
"$B" filter temp_sensor.csv --q 0.005  --r 0.04 --smooth
"$B" filter altitude.csv    --filter cv --q 0.0025 --r 1.0 --smooth

# --- 3. Side-by-side comparison of the three estimators ---
"$B" compare step.csv --q 0.1 --r 0.25 > compare_step.txt
"$B" compare ramp.csv --q 0.01 --r 0.25 > compare_ramp.txt

# --- 4. Adaptive noise estimation on the temperature sensor ---
"$B" filter temp_sensor.csv --q 0.01 --r 0.04 --adaptive --output temp_sensor_adaptive.filtered.csv > adaptive_temp.txt

# --- 5. Batch mode over all five inputs (scalar filter, one Q/R) ---
mkdir -p batch_input
cp step.csv sine.csv ramp.csv temp_sensor.csv altitude.csv batch_input/
"$B" batch batch_input --output-dir batch_output --q 0.1 --r 0.25 > batch_summary.txt
rm -rf batch_input

# --- 6. Hand-checkable example (computed live by the filter) ---
"$B" handcheck > hand_check.txt

echo "examples regenerated."
