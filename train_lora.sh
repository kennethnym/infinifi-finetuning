#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: train_lora.sh [options]

Train a condition-gated LoRA adapter for facebook/musicgen-small.

Options:
  --rank N                LoRA rank (default: 8; use 8 or 16 for the experiment)
  --alpha N               LoRA alpha (default: same as rank)
  --adapter-dropout RATE  LoRA input dropout in [0, 1) (default: 0.05)
  --batch-size N          Batch size per GPU (default: 2)
  --epochs N              Number of epochs (default: 3)
  --updates-per-epoch N   Optimizer updates per epoch (default: 500)
  --segment-duration SEC  Training segment duration (default: 20)
  --num-workers N         Data-loader workers (default: 4)
  --lr RATE               AdamW learning rate (default: 1e-4)
  --warmup-steps N        Cosine warmup updates (default: 5% of all updates)
  --valid-samples N       Validation samples per epoch (default: 128)
  --evaluate-samples N    Evaluation samples (default: 128)
  --generate-samples N    Generated monitoring samples (default: 4)
  --generate-every N      Generate monitoring samples every N epochs (default: 1)
  --checkpoint-every N    Save a checkpoint every N epochs (default: 1)
  --seed N                Training seed (default: 2036)
  -h, --help              Show this help
EOF
}

fail() {
    printf 'train_lora.sh: %s\n' "$1" >&2
    exit 2
}

require_option_value() {
    local option="$1"
    if (( $# < 2 )) || [[ "$2" == --* ]]; then
        fail "${option} requires a value"
    fi
}

require_positive_integer() {
    local option="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
        fail "${option} must be a positive integer"
}

require_nonnegative_integer() {
    local option="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] ||
        fail "${option} must be a non-negative integer"
}

require_positive_number() {
    local option="$1"
    local value="$2"
    if [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
        local mantissa="${BASH_REMATCH[1]}"
        [[ -n "${mantissa//[.0]/}" ]] && return
    fi
    fail "${option} must be a positive number"
}

require_adapter_dropout() {
    local value="$1"
    [[ "$value" =~ ^(0([.][0-9]*)?|[.][0-9]+)$ ]] ||
        fail "--adapter-dropout must be a number greater than or equal to 0 and lower than 1"
}

lora_rank=8
lora_alpha=
lora_dropout=0.05
lora_batch_size=2
lora_epochs=3
lora_updates_per_epoch=500
lora_segment_duration=20
lora_num_workers=4
lora_lr=1e-4
lora_warmup_steps=
lora_valid_samples=128
lora_evaluate_samples=128
lora_generate_samples=4
lora_generate_every=1
lora_checkpoint_every=1
lora_seed=2036

while (( $# > 0 )); do
    case "$1" in
        --rank)
            require_option_value "$@"
            lora_rank="$2"
            shift 2
            ;;
        --alpha)
            require_option_value "$@"
            lora_alpha="$2"
            shift 2
            ;;
        --adapter-dropout)
            require_option_value "$@"
            lora_dropout="$2"
            shift 2
            ;;
        --batch-size)
            require_option_value "$@"
            lora_batch_size="$2"
            shift 2
            ;;
        --epochs)
            require_option_value "$@"
            lora_epochs="$2"
            shift 2
            ;;
        --updates-per-epoch)
            require_option_value "$@"
            lora_updates_per_epoch="$2"
            shift 2
            ;;
        --segment-duration)
            require_option_value "$@"
            lora_segment_duration="$2"
            shift 2
            ;;
        --num-workers)
            require_option_value "$@"
            lora_num_workers="$2"
            shift 2
            ;;
        --lr)
            require_option_value "$@"
            lora_lr="$2"
            shift 2
            ;;
        --warmup-steps)
            require_option_value "$@"
            lora_warmup_steps="$2"
            shift 2
            ;;
        --valid-samples)
            require_option_value "$@"
            lora_valid_samples="$2"
            shift 2
            ;;
        --evaluate-samples)
            require_option_value "$@"
            lora_evaluate_samples="$2"
            shift 2
            ;;
        --generate-samples)
            require_option_value "$@"
            lora_generate_samples="$2"
            shift 2
            ;;
        --generate-every)
            require_option_value "$@"
            lora_generate_every="$2"
            shift 2
            ;;
        --checkpoint-every)
            require_option_value "$@"
            lora_checkpoint_every="$2"
            shift 2
            ;;
        --seed)
            require_option_value "$@"
            lora_seed="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

require_positive_integer --rank "$lora_rank"
if [[ -z "$lora_alpha" ]]; then
    lora_alpha="$lora_rank"
fi
require_positive_number --alpha "$lora_alpha"
require_adapter_dropout "$lora_dropout"
require_positive_integer --batch-size "$lora_batch_size"
require_positive_integer --epochs "$lora_epochs"
require_positive_integer --updates-per-epoch "$lora_updates_per_epoch"
require_positive_number --segment-duration "$lora_segment_duration"
require_nonnegative_integer --num-workers "$lora_num_workers"
require_positive_number --lr "$lora_lr"
require_positive_integer --valid-samples "$lora_valid_samples"
require_positive_integer --evaluate-samples "$lora_evaluate_samples"
require_positive_integer --generate-samples "$lora_generate_samples"
require_positive_integer --generate-every "$lora_generate_every"
require_positive_integer --checkpoint-every "$lora_checkpoint_every"
require_nonnegative_integer --seed "$lora_seed"

lora_total_updates=$((lora_epochs * lora_updates_per_epoch))
if [[ -z "$lora_warmup_steps" ]]; then
    lora_warmup_steps=$(((lora_total_updates + 19) / 20))
    if (( lora_warmup_steps >= lora_total_updates )); then
        lora_warmup_steps=$((lora_total_updates - 1))
    fi
fi
require_nonnegative_integer --warmup-steps "$lora_warmup_steps"
(( lora_warmup_steps < lora_total_updates )) ||
    fail "--warmup-steps must be lower than the total number of updates"

exec dora \
    -P audiocraft \
    run \
    solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small \
    conditioner=text2music \
    dset=audio/lofi \
    "seed=${lora_seed}" \
    "dataset.num_workers=${lora_num_workers}" \
    "dataset.batch_size=${lora_batch_size}" \
    "dataset.segment_duration=${lora_segment_duration}" \
    dataset.sample_on_weight=false \
    dataset.sample_on_duration=false \
    dataset.train.permutation_on_files=true \
    dataset.train.merge_text_p=0 \
    dataset.train.drop_desc_p=0 \
    dataset.train.drop_other_p=0 \
    "dataset.valid.num_samples=${lora_valid_samples}" \
    "dataset.evaluate.num_samples=${lora_evaluate_samples}" \
    "dataset.generate.num_samples=${lora_generate_samples}" \
    "generate.every=${lora_generate_every}" \
    generate.lm.prompted_samples=false \
    "optim.epochs=${lora_epochs}" \
    "optim.updates_per_epoch=${lora_updates_per_epoch}" \
    optim.optimizer=adamw \
    "optim.lr=${lora_lr}" \
    "optim.adam.betas=[0.9,0.95]" \
    optim.adam.weight_decay=0.01 \
    optim.ema.use=false \
    schedule.lr_scheduler=cosine \
    "schedule.cosine.warmup=${lora_warmup_steps}" \
    schedule.cosine.lr_min_ratio=0.1 \
    "checkpoint.save_every=${lora_checkpoint_every}" \
    conditioners.description.t5.word_dropout=0 \
    classifier_free_guidance.training_dropout=0 \
    transformer_lm.lora.enabled=true \
    "transformer_lm.lora.rank=${lora_rank}" \
    "transformer_lm.lora.alpha=${lora_alpha}" \
    "transformer_lm.lora.dropout=${lora_dropout}"
