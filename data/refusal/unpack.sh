#!/bin/bash
# Unpack the refusal train/test data into falsereject_data/, where
# configs/qwen_qwen_refusal_combined_judge_freeform_natural*.yaml read it, and verify the checksums.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p falsereject_data
for s in train test; do
  f=combined_qwen3_8b_system_prompt_${s}_judge_natural.jsonl
  gunzip -c "data/refusal/$f.gz" > "falsereject_data/$f"
done
cd falsereject_data
sha256sum -c <<'SUMS'
0f5e402ed5f968d051fc7778b15e64808554b8b843a0a064c274cf8e1f7a72d4  combined_qwen3_8b_system_prompt_train_judge_natural.jsonl
d89fe95bdfa3398d5580a9ec891328ff700fcb597f5ca204ba49d0b8f3aa9d05  combined_qwen3_8b_system_prompt_test_judge_natural.jsonl
SUMS
