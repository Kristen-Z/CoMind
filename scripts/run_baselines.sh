#!/usr/bin/env bash
# Run the API baselines (GPT, Claude, Gemini) on all tasks. On the train split, also evaluate.
# Usage: COMIND_DATA_ROOT=/path/to/comind bash scripts/run_baselines.sh [test|train]
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${1:-test}

run() {  # run <backend> <model>
    for task in joint_attention handover scoia; do
        python -m comind.infer --task "$task" --backend "$1" --model "$2" --split "$SPLIT" \
            --output_dir "results/$SPLIT/$task/$1"
        if [[ $SPLIT == train ]]; then
            python -m comind.evaluate --task "$task" --split train --pred_dir "results/$SPLIT/$task/$1"
        fi
    done
}

run openai gpt-5.1
run claude claude-opus-4-6
run gemini gemini-3-flash-preview
