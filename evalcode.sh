#!/bin/bash
# Usage: bash evalcode.sh <generated_dir> [split]
# Example:
#   bash evalcode.sh eval_results/20260417_033050_test/qwen3.5-122b/with_background test

GENERATED_DIR=${1:?'Usage: bash evalcode.sh <generated_dir> [split]'}
SPLIT=${2:-test}
RUN_DIR="$(dirname "$(dirname "$GENERATED_DIR")")"

uv run python eval/scripts/test_generated_code.py \
  --generated-dir "$GENERATED_DIR" \
  --split "$SPLIT" \
  --output-dir "$RUN_DIR"
