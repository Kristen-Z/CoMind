#!/usr/bin/env bash
# Run a Qwen-VL baseline on all tasks. On the train split, also evaluate.
# Usage: COMIND_DATA_ROOT=/path/to/comind sbatch scripts/slurm_qwen.sbatch [Qwen/Qwen3-VL-8B-Instruct] [test|train]
#SBATCH --job-name=comind_qwen
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
MODEL=${1:-Qwen/Qwen3-VL-8B-Instruct}
SPLIT=${2:-test}
for task in joint_attention handover scoia; do
    out="results/$SPLIT/$task/$(basename "$MODEL")"
    python -m comind.infer --task "$task" --backend qwen --model "$MODEL" --split "$SPLIT" --output_dir "$out"
    if [[ $SPLIT == train ]]; then
        python -m comind.evaluate --task "$task" --split train --pred_dir "$out"
    fi
done
