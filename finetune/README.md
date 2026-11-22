# Fine-tuning Qwen3-VL on CoMind

LoRA fine-tuning of Qwen3-VL 8B and 32B on the three CoMind tasks, with the settings of the models marked FT in the paper. The training code in [`qwenvl/`](qwenvl) is `qwen-vl-finetune` from [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (commit `9658872`) with a few [changes](#changes-to-qwen-vl-finetune).

```
finetune/
├── prepare_data.py    # CoMind train split -> Qwen-VL conversations
├── train.sh           # LoRA fine-tuning of one model on one task
├── infer.py           # predictions of a fine-tuned model, as written by comind.infer
├── configs/           # DeepSpeed ZeRO-2 (8B) and ZeRO-3 (32B)
└── qwenvl/            # training code from Qwen3-VL
```

## Setup

The paper's models were trained with Python 3.11, CUDA 12.8 and the versions in [`requirements.txt`](requirements.txt). From the repository root:

```bash
pip install -e .
pip install -r finetune/requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

FlashAttention 2 is required: training packs the samples of a batch into one sequence (`--data_flatten`) and uses its variable-length kernel. DeepSpeed needs the CUDA toolkit, not only the driver: if training stops with `CUDA_HOME does not exist`, load your cluster's CUDA module or set `CUDA_HOME` to the toolkit directory.

## Training data

Download and prepare the train split as described in the [main README](../README.md#data), then write the training conversations:

```bash
export COMIND_DATA_ROOT=data/comind
python -m comind.prepare --split train
python finetune/prepare_data.py
```

Each annotated sample becomes one conversation: the model gets the inputs of `comind.infer` (the reference frame, for handover and SCOIA after the 5 context frames, and the transcript) with the [benchmark prompt](../comind/prompts) of the task, and the answer is the annotation in the JSON format the prompt asks for, with boxes as `[ymin, xmin, ymax, xmax]` normalised to [0, 1000]. Samples whose annotation lacks a field of the answer are left out: handover samples marked `skip` or without a box, SCOIA samples without a cue type or box, and joint attention samples without a cue type or without a box in each view (of several boxes in a view, the largest is used).

This writes `$COMIND_DATA_ROOT/finetune/<task>_train.json` and the context frames next to it: 5,574 joint attention, 309 handover and 985 SCOIA samples.

## Training

```bash
bash finetune/train.sh scoia 8b      # <joint_attention|handover|scoia> <8b|32b>
```

| | |
|---|---|
| LoRA | rank 64, alpha 128, dropout 0.05, on the q/k/v/o projections of the language model; the vision encoder and merger are frozen |
| Optimisation | AdamW, learning rate 2e-4 with cosine decay and 3 % warm-up, weight decay 0.01, gradient clipping at 1, bf16 |
| Batch | 8 samples per step (2 per GPU, gradient accumulation for the rest) |
| Epochs | 5 |
| Images | at most 401,408 pixels each: 392 visual tokens for a 1408×704 frame, 361 for 1408×1408 |
| DeepSpeed | ZeRO-2 for 8B, ZeRO-3 for 32B |

The paper's runs used two 96 GB GPUs each; with 1 or 4 GPUs, `train.sh` adjusts the gradient accumulation to keep 8 samples per step. The LoRA adapter is written to `checkpoints/qwen3_vl_<size>_<task>/` (set `OUTPUT_DIR` to change it), with intermediate checkpoints every 500 steps for 8B (all kept) and every 50 steps for 32B (the last 3 kept). An interrupted run resumes from its last checkpoint. Arguments after the model size are passed on to `qwenvl/train/train_qwen.py`, e.g. `--save_steps 100`. Set `REPORT_TO=wandb` to log to Weights & Biases.

Five epochs are 3,485 steps for joint attention, 195 for handover and 620 for SCOIA. The paper's models were trained on an earlier version of the train annotations:

| Task | Training samples in the paper | Steps (5 epochs) | 8B checkpoint | 32B checkpoint |
|---|---|---|---|---|
| Joint attention | 5,523 | 3,455 | final | step 2,650 |
| Handover | 214 | 135 | final | final |
| SCOIA | 937 | 590 | final | final |

To keep step 2,650 of a 32B joint attention run, add `--save_steps 2650`.

## Inference and evaluation

```bash
python finetune/infer.py --task scoia --adapter checkpoints/qwen3_vl_8b_scoia
```

`infer.py` writes the same files as `python -m comind.infer` to `results/<task>/<adapter name>/`, with the inputs of training: the same prompt, the context frames as separate images before the reference frame, and the same image size. Fine-tuned models answer with boxes in the format of the prompt (`yxyx_1000`), where the Qwen3-VL base models of `comind.infer` answer in their native `[xmin, ymin, xmax, ymax]` order; `run_config.json` records the format for the evaluation. The base model is read from the adapter; `--adapter` also takes a checkpoint directory, and without it the base model gets the same inputs. Submit test predictions through the [leaderboard](https://comind.ethz.ch/#leaderboards), or evaluate on the train split:

```bash
python finetune/infer.py --task scoia --adapter checkpoints/qwen3_vl_8b_scoia --split train
python -m comind.evaluate --task scoia --split train --pred_dir results/scoia/qwen3_vl_8b_scoia
```

`comind.evaluate` and `comind.visualize` read the box format of the predictions from their `run_config.json`.

## Changes to qwen-vl-finetune

- `qwenvl/data/__init__.py`: `--dataset_use` takes paths of annotation files instead of names of the example datasets.
- `qwenvl/train/train_qwen.py`: reentrant gradient checkpointing, for ZeRO-3.
- `qwenvl/train/trainer.py`: `return_mask` takes the arguments of `create_causal_mask` in transformers 4.57.
- `train.sh` replaces the example scripts. Of the arguments of the paper's runs, it leaves out `--tune_mm_vision`, `--tune_mm_mlp`, `--tune_mm_llm` and `--per_device_eval_batch_size`, which have no effect with `--lora_enable` and without evaluation.
- The demo data and the tools are not included. `configs/` holds the unchanged DeepSpeed configurations.

## License

The code in this directory is released under the [Apache License 2.0](LICENSE), the license of Qwen3-VL. `qwenvl/train/train_qwen.py` is further based on [FastChat](https://github.com/lm-sys/FastChat) and [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), both Apache 2.0.
