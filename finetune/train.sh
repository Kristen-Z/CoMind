#!/usr/bin/env bash
# LoRA fine-tuning of Qwen3-VL on one CoMind task, with the settings of the paper.
# Usage: COMIND_DATA_ROOT=/path/to/comind bash finetune/train.sh <joint_attention|handover|scoia> [8b|32b] [args]
# Reads $COMIND_DATA_ROOT/finetune/<task>_train.json (written by prepare_data.py) and writes the adapter to
# checkpoints/qwen3_vl_<size>_<task> (or $OUTPUT_DIR). Further arguments are passed to train_qwen.py.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:?"usage: $0 <joint_attention|handover|scoia> [8b|32b] [train_qwen.py arguments]"}
SIZE=${2:-8b}
shift $(( $# < 2 ? $# : 2 ))

case $SIZE in
    8b)  MODEL=Qwen/Qwen3-VL-8B-Instruct  DEEPSPEED=zero2.json SAVE_STEPS=500 SAVE_LIMIT=100 WORKERS=4 ;;
    # ZeRO-3 shards the 32B weights, so that training fits on two 96 GB GPUs
    32b) MODEL=Qwen/Qwen3-VL-32B-Instruct DEEPSPEED=zero3.json SAVE_STEPS=50  SAVE_LIMIT=3   WORKERS=2 ;;
    *) echo "Unknown model size '$SIZE', use 8b or 32b." >&2; exit 1 ;;
esac
DATA=${COMIND_DATA_ROOT:?"set COMIND_DATA_ROOT"}/finetune/${TASK}_train.json
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/qwen3_vl_${SIZE}_${TASK}}

# 8 samples per optimizer step, as in the paper (2 GPUs x 2 samples x 2 accumulation steps)
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
BATCH_SIZE=2
if (( NUM_GPUS < 1 || 8 % (BATCH_SIZE * NUM_GPUS) )); then
    echo "Cannot split 8 samples per step over $NUM_GPUS GPUs with $BATCH_SIZE samples each." >&2; exit 1
fi

python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}" \
    finetune/qwenvl/train/train_qwen.py \
    --model_name_or_path "$MODEL" \
    --dataset_use "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "finetune/configs/$DEEPSPEED" \
    --bf16 \
    --data_flatten True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_train_epochs 5 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps $(( 8 / (BATCH_SIZE * NUM_GPUS) )) \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1 \
    --max_pixels 401408 \
    --min_pixels 784 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --logging_steps 1 \
    --dataloader_num_workers "$WORKERS" \
    --run_name "qwen3_vl_${SIZE}_${TASK}" \
    --report_to "${REPORT_TO:-none}" \
    "$@"
