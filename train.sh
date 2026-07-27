#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: train.sh [options]

Fine-tune facebook/musicgen-small on the prepared lo-fi dataset.

Options:
  --batch-size N          Batch size per GPU (default: 2)
  --epochs N              Number of epochs (default: 10)
  --updates-per-epoch N   Optimizer updates per epoch (default: 500)
  --segment-duration SEC  Training segment duration (default: 20)
  --num-workers N         Data-loader workers (default: 4)
  --lr RATE               AdamW learning rate (default: 1e-5)
  --warmup-steps N        Cosine warmup updates (default: 5% of all updates)
  --train-samples N       Samples per epoch for random file sampling
                           (default: batch size x updates)
  --random-file-sampling  Sample files uniformly with replacement instead of
                           using the default deterministic file permutation
  --valid-samples N       Validation samples per epoch (default: 128)
  --evaluate-samples N    Evaluation samples (default: 128)
  --generate-samples N    Generated monitoring samples (default: 4)
  --generate-every N      Generate monitoring samples every N epochs (default: 5)
  --checkpoint-every N    Save a checkpoint every N epochs (default: 5)
  --word-dropout RATE     T5 word-dropout probability in [0, 1]
                           (default: AudioCraft configuration)
  --cfg-dropout RATE      Classifier-free training dropout in [0, 1]
                           (default: AudioCraft configuration)
  --merge-text-p RATE     Metadata-merge probability in [0, 1]
                           (default: AudioCraft configuration)
  --drop-desc-p RATE      Description-drop probability on merge in [0, 1]
                           (default: AudioCraft configuration)
  --drop-other-p RATE     Metadata-field dropout probability in [0, 1]
                           (default: AudioCraft configuration)
  -h, --help              Show this help
EOF
}

fail() {
    printf 'train.sh: %s\n' "$1" >&2
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

require_probability() {
    local option="$1"
    local value="$2"
    [[ "$value" =~ ^(0([.][0-9]*)?|[.][0-9]+|1([.]0*)?)$ ]] ||
        fail "${option} must be a number between 0 and 1"
}

finetune_batch_size=2
finetune_epochs=10
finetune_updates_per_epoch=500
finetune_segment_duration=20
finetune_num_workers=4
finetune_lr=1e-5
finetune_warmup_steps=
finetune_train_samples=
finetune_file_permutation=true
finetune_valid_samples=128
finetune_evaluate_samples=128
finetune_generate_samples=4
finetune_generate_every=5
finetune_checkpoint_every=5
finetune_word_dropout=
finetune_cfg_dropout=
finetune_merge_text_p=
finetune_drop_desc_p=
finetune_drop_other_p=

while (( $# > 0 )); do
    case "$1" in
        --batch-size)
            require_option_value "$@"
            finetune_batch_size="$2"
            shift 2
            ;;
        --epochs)
            require_option_value "$@"
            finetune_epochs="$2"
            shift 2
            ;;
        --updates-per-epoch)
            require_option_value "$@"
            finetune_updates_per_epoch="$2"
            shift 2
            ;;
        --segment-duration)
            require_option_value "$@"
            finetune_segment_duration="$2"
            shift 2
            ;;
        --num-workers)
            require_option_value "$@"
            finetune_num_workers="$2"
            shift 2
            ;;
        --lr)
            require_option_value "$@"
            finetune_lr="$2"
            shift 2
            ;;
        --warmup-steps)
            require_option_value "$@"
            finetune_warmup_steps="$2"
            shift 2
            ;;
        --train-samples)
            require_option_value "$@"
            finetune_train_samples="$2"
            shift 2
            ;;
        --random-file-sampling)
            finetune_file_permutation=false
            shift
            ;;
        --valid-samples)
            require_option_value "$@"
            finetune_valid_samples="$2"
            shift 2
            ;;
        --evaluate-samples)
            require_option_value "$@"
            finetune_evaluate_samples="$2"
            shift 2
            ;;
        --generate-samples)
            require_option_value "$@"
            finetune_generate_samples="$2"
            shift 2
            ;;
        --generate-every)
            require_option_value "$@"
            finetune_generate_every="$2"
            shift 2
            ;;
        --checkpoint-every)
            require_option_value "$@"
            finetune_checkpoint_every="$2"
            shift 2
            ;;
        --word-dropout)
            require_option_value "$@"
            finetune_word_dropout="$2"
            shift 2
            ;;
        --cfg-dropout)
            require_option_value "$@"
            finetune_cfg_dropout="$2"
            shift 2
            ;;
        --merge-text-p)
            require_option_value "$@"
            finetune_merge_text_p="$2"
            shift 2
            ;;
        --drop-desc-p)
            require_option_value "$@"
            finetune_drop_desc_p="$2"
            shift 2
            ;;
        --drop-other-p)
            require_option_value "$@"
            finetune_drop_other_p="$2"
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

require_positive_integer --batch-size "$finetune_batch_size"
require_positive_integer --epochs "$finetune_epochs"
require_positive_integer --updates-per-epoch "$finetune_updates_per_epoch"
require_positive_number --segment-duration "$finetune_segment_duration"
require_nonnegative_integer --num-workers "$finetune_num_workers"
require_positive_number --lr "$finetune_lr"
require_positive_integer --valid-samples "$finetune_valid_samples"
require_positive_integer --evaluate-samples "$finetune_evaluate_samples"
require_positive_integer --generate-samples "$finetune_generate_samples"
require_positive_integer --generate-every "$finetune_generate_every"
require_positive_integer --checkpoint-every "$finetune_checkpoint_every"

if [[ -n "$finetune_word_dropout" ]]; then
    require_probability --word-dropout "$finetune_word_dropout"
fi
if [[ -n "$finetune_cfg_dropout" ]]; then
    require_probability --cfg-dropout "$finetune_cfg_dropout"
fi
if [[ -n "$finetune_merge_text_p" ]]; then
    require_probability --merge-text-p "$finetune_merge_text_p"
fi
if [[ -n "$finetune_drop_desc_p" ]]; then
    require_probability --drop-desc-p "$finetune_drop_desc_p"
fi
if [[ -n "$finetune_drop_other_p" ]]; then
    require_probability --drop-other-p "$finetune_drop_other_p"
fi

finetune_total_updates=$((finetune_epochs * finetune_updates_per_epoch))
if [[ -z "$finetune_warmup_steps" ]]; then
    finetune_warmup_steps=$(((finetune_total_updates + 19) / 20))
    if (( finetune_warmup_steps >= finetune_total_updates )); then
        finetune_warmup_steps=$((finetune_total_updates - 1))
    fi
fi
require_nonnegative_integer --warmup-steps "$finetune_warmup_steps"
(( finetune_warmup_steps < finetune_total_updates )) ||
    fail "--warmup-steps must be lower than the total number of updates"

if [[ -z "$finetune_train_samples" ]]; then
    finetune_train_samples=$((finetune_batch_size * finetune_updates_per_epoch))
fi
require_positive_integer --train-samples "$finetune_train_samples"

dora_args=(
    -P audiocraft
    run
    solver=musicgen/musicgen_base_32khz
    model/lm/model_scale=small
    continue_from=//pretrained/facebook/musicgen-small
    conditioner=text2music
    dset=audio/lofi
    "dataset.num_workers=${finetune_num_workers}"
    "dataset.batch_size=${finetune_batch_size}"
    "dataset.segment_duration=${finetune_segment_duration}"
    "dataset.train.num_samples=${finetune_train_samples}"
    dataset.sample_on_weight=false
    dataset.sample_on_duration=false
    "dataset.permutation_on_files=${finetune_file_permutation}"
    "dataset.valid.num_samples=${finetune_valid_samples}"
    "dataset.evaluate.num_samples=${finetune_evaluate_samples}"
    "dataset.generate.num_samples=${finetune_generate_samples}"
    "generate.every=${finetune_generate_every}"
    generate.lm.prompted_samples=false
    "optim.epochs=${finetune_epochs}"
    "optim.updates_per_epoch=${finetune_updates_per_epoch}"
    optim.optimizer=adamw
    "optim.lr=${finetune_lr}"
    "optim.adam.betas=[0.9,0.95]"
    optim.adam.weight_decay=0.01
    optim.ema.use=true
    optim.ema.device=cpu
    optim.ema.updates=10
    schedule.lr_scheduler=cosine
    "schedule.cosine.warmup=${finetune_warmup_steps}"
    schedule.cosine.lr_min_ratio=0.1
    "checkpoint.save_every=${finetune_checkpoint_every}"
)

if [[ -n "$finetune_word_dropout" ]]; then
    dora_args+=(
        "conditioners.description.t5.word_dropout=${finetune_word_dropout}"
    )
fi
if [[ -n "$finetune_cfg_dropout" ]]; then
    dora_args+=(
        "classifier_free_guidance.training_dropout=${finetune_cfg_dropout}"
    )
fi
if [[ -n "$finetune_merge_text_p" ]]; then
    dora_args+=("dataset.train.merge_text_p=${finetune_merge_text_p}")
fi
if [[ -n "$finetune_drop_desc_p" ]]; then
    dora_args+=("dataset.train.drop_desc_p=${finetune_drop_desc_p}")
fi
if [[ -n "$finetune_drop_other_p" ]]; then
    dora_args+=("dataset.train.drop_other_p=${finetune_drop_other_p}")
fi

exec dora "${dora_args[@]}"
