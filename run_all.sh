#!/usr/bin/env bash
# Run the full experiment end-to-end. Usage: ./run_all.sh [validate|test]
set -euo pipefail
cd "$(dirname "$0")"
SPLIT="${1:-test}"
PY="${PYTHON:-python}"

if [ ! -f data/raw/train.csv ]; then
  echo ">> No data found - generating synthetic placeholder data"
  $PY scripts/generate_synthetic_data.py
fi

$PY scripts/audit_dataset.py
$PY scripts/clean_dataset.py
$PY scripts/build_raw_index.py
$PY scripts/build_clean_index.py
$PY scripts/evaluate_raw.py --split "$SPLIT"
$PY scripts/evaluate_cleaned.py --split "$SPLIT"
$PY scripts/compare_results.py --split "$SPLIT"
