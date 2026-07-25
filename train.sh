#!/usr/bin/env bash
set -euo pipefail

# Single-GPU fine-tuning defaults: 10 epochs x 500 updates = 5,000 updates.
# Override any value from the environment for shorter tests or larger runs.
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
FINETUNE_UPDATES_PER_EPOCH="${FINETUNE_UPDATES_PER_EPOCH:-500}"
FINETUNE_SEGMENT_DURATION="${FINETUNE_SEGMENT_DURATION:-20}"
FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-4}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"

finetune_total_updates=$((FINETUNE_EPOCHS * FINETUNE_UPDATES_PER_EPOCH))
finetune_default_warmup=$(((finetune_total_updates + 19) / 20))
finetune_default_train_samples=$((FINETUNE_BATCH_SIZE * FINETUNE_UPDATES_PER_EPOCH))

FINETUNE_WARMUP_STEPS="${FINETUNE_WARMUP_STEPS:-${finetune_default_warmup}}"
FINETUNE_TRAIN_SAMPLES="${FINETUNE_TRAIN_SAMPLES:-${finetune_default_train_samples}}"
FINETUNE_VALID_SAMPLES="${FINETUNE_VALID_SAMPLES:-128}"
FINETUNE_EVALUATE_SAMPLES="${FINETUNE_EVALUATE_SAMPLES:-128}"
FINETUNE_GENERATE_SAMPLES="${FINETUNE_GENERATE_SAMPLES:-4}"

dora_args=(
    -P audiocraft
    run
    solver=musicgen/musicgen_base_32khz
    model/lm/model_scale=small
    continue_from=//pretrained/facebook/musicgen-small
    conditioner=text2music
    dset=audio/lofi
    "dataset.num_workers=${FINETUNE_NUM_WORKERS}"
    "dataset.batch_size=${FINETUNE_BATCH_SIZE}"
    "dataset.segment_duration=${FINETUNE_SEGMENT_DURATION}"
    "dataset.train.num_samples=${FINETUNE_TRAIN_SAMPLES}"
    "dataset.valid.num_samples=${FINETUNE_VALID_SAMPLES}"
    "dataset.evaluate.num_samples=${FINETUNE_EVALUATE_SAMPLES}"
    "dataset.generate.num_samples=${FINETUNE_GENERATE_SAMPLES}"
    generate.every=5
    generate.lm.prompted_samples=false
    "optim.epochs=${FINETUNE_EPOCHS}"
    "optim.updates_per_epoch=${FINETUNE_UPDATES_PER_EPOCH}"
    optim.optimizer=adamw
    "optim.lr=${FINETUNE_LR}"
    "optim.adam.betas=[0.9,0.95]"
    optim.adam.weight_decay=0.01
    optim.ema.use=true
    optim.ema.device=cpu
    optim.ema.updates=10
    schedule.lr_scheduler=cosine
    "schedule.cosine.warmup=${FINETUNE_WARMUP_STEPS}"
    schedule.cosine.lr_min_ratio=0.1
    checkpoint.save_every=5
)

exec dora "${dora_args[@]}"
