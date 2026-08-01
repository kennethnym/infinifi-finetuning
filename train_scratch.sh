#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: train_scratch.sh [options]

Distill a compact MusicGen-compatible lo-fi LM from random initialization.
The EnCodec codec and T5 backbone remain pretrained and frozen.

Options:
  --teacher MODEL              Teacher (default: facebook/musicgen-large)
  --temperature RATE           KD temperature (default: 2)
  --initial-kd-weight RATE     Initial teacher KL weight (default: 0.5)
  --initial-ce-weight RATE     Initial hard-token CE weight (default: 0.5)
  --kd-weight RATE             Final teacher KL weight (default: 0.75)
  --ce-weight RATE             Final hard-token CE weight (default: 0.25)
  --weight-transition-updates N  Linear loss transition length (default: 10000)
  --conditional-only           Do not distill the unconditional CFG branch
  --dim N                      Transformer width (default: 640)
  --heads N                    Attention heads (default: 10)
  --layers N                   Transformer layers (default: 10)
  --batch-size N               Microbatch size per GPU (default: 1)
  --grad-accumulation N        Microbatches per optimizer update (default: 8)
  --epochs N                   Number of epochs (default: 20)
  --updates-per-epoch N        Optimizer updates per epoch (default: 1000)
  --segment-duration SEC       Random crop duration (default: 10)
  --num-workers N              Data-loader workers (default: 4)
  --lr RATE                    AdamW learning rate (default: 3e-4)
  --warmup-steps N             Cosine warmup optimizer updates (default: 5%)
  --valid-samples N            Validation samples per epoch (default: 128)
  --evaluate-samples N         Evaluation samples (default: 128)
  --generate-samples N         Generated monitoring samples (default: 4)
  --generate-every N           Generate every N epochs (default: 5)
  --checkpoint-every N         Save every N epochs (default: 1)
  --continue-from CHECKPOINT    Initialize from an earlier student checkpoint
  --train-data PATH             Override the AudioCraft training manifest directory
  --valid-data PATH             Override validation/evaluation manifest directory
  --seed N                      Training seed (default: 2036)
  -h, --help                    Show this help
EOF
}

fail() {
    printf 'train_scratch.sh: %s\n' "$1" >&2
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

require_nonnegative_number() {
    local option="$1"
    local value="$2"
    [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] ||
        fail "${option} must be a non-negative number"
}

scratch_teacher=facebook/musicgen-large
scratch_temperature=2
scratch_initial_kd_weight=0.5
scratch_initial_ce_weight=0.5
scratch_kd_weight=0.75
scratch_ce_weight=0.25
scratch_weight_transition_updates=10000
scratch_cfg_branches=true
scratch_dim=640
scratch_heads=10
scratch_layers=10
scratch_batch_size=1
scratch_grad_accumulation=8
scratch_epochs=20
scratch_updates_per_epoch=1000
scratch_segment_duration=10
scratch_num_workers=4
scratch_lr=3e-4
scratch_warmup_steps=
scratch_valid_samples=128
scratch_evaluate_samples=128
scratch_generate_samples=4
scratch_generate_every=5
scratch_checkpoint_every=1
scratch_continue_from=
scratch_train_data=
scratch_valid_data=
scratch_seed=2036

while (( $# > 0 )); do
    case "$1" in
        --teacher)
            require_option_value "$@"
            scratch_teacher="$2"
            shift 2
            ;;
        --temperature)
            require_option_value "$@"
            scratch_temperature="$2"
            shift 2
            ;;
        --initial-kd-weight)
            require_option_value "$@"
            scratch_initial_kd_weight="$2"
            shift 2
            ;;
        --initial-ce-weight)
            require_option_value "$@"
            scratch_initial_ce_weight="$2"
            shift 2
            ;;
        --kd-weight)
            require_option_value "$@"
            scratch_kd_weight="$2"
            shift 2
            ;;
        --ce-weight)
            require_option_value "$@"
            scratch_ce_weight="$2"
            shift 2
            ;;
        --weight-transition-updates)
            require_option_value "$@"
            scratch_weight_transition_updates="$2"
            shift 2
            ;;
        --conditional-only)
            scratch_cfg_branches=false
            shift
            ;;
        --dim)
            require_option_value "$@"
            scratch_dim="$2"
            shift 2
            ;;
        --heads)
            require_option_value "$@"
            scratch_heads="$2"
            shift 2
            ;;
        --layers)
            require_option_value "$@"
            scratch_layers="$2"
            shift 2
            ;;
        --batch-size)
            require_option_value "$@"
            scratch_batch_size="$2"
            shift 2
            ;;
        --grad-accumulation)
            require_option_value "$@"
            scratch_grad_accumulation="$2"
            shift 2
            ;;
        --epochs)
            require_option_value "$@"
            scratch_epochs="$2"
            shift 2
            ;;
        --updates-per-epoch)
            require_option_value "$@"
            scratch_updates_per_epoch="$2"
            shift 2
            ;;
        --segment-duration)
            require_option_value "$@"
            scratch_segment_duration="$2"
            shift 2
            ;;
        --num-workers)
            require_option_value "$@"
            scratch_num_workers="$2"
            shift 2
            ;;
        --lr)
            require_option_value "$@"
            scratch_lr="$2"
            shift 2
            ;;
        --warmup-steps)
            require_option_value "$@"
            scratch_warmup_steps="$2"
            shift 2
            ;;
        --valid-samples)
            require_option_value "$@"
            scratch_valid_samples="$2"
            shift 2
            ;;
        --evaluate-samples)
            require_option_value "$@"
            scratch_evaluate_samples="$2"
            shift 2
            ;;
        --generate-samples)
            require_option_value "$@"
            scratch_generate_samples="$2"
            shift 2
            ;;
        --generate-every)
            require_option_value "$@"
            scratch_generate_every="$2"
            shift 2
            ;;
        --checkpoint-every)
            require_option_value "$@"
            scratch_checkpoint_every="$2"
            shift 2
            ;;
        --continue-from)
            require_option_value "$@"
            scratch_continue_from="$2"
            shift 2
            ;;
        --train-data)
            require_option_value "$@"
            scratch_train_data="$2"
            shift 2
            ;;
        --valid-data)
            require_option_value "$@"
            scratch_valid_data="$2"
            shift 2
            ;;
        --seed)
            require_option_value "$@"
            scratch_seed="$2"
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

[[ -n "${scratch_teacher//[[:space:]]/}" ]] || fail "--teacher must be non-empty"
require_positive_number --temperature "$scratch_temperature"
require_nonnegative_number --initial-kd-weight "$scratch_initial_kd_weight"
require_nonnegative_number --initial-ce-weight "$scratch_initial_ce_weight"
require_nonnegative_number --kd-weight "$scratch_kd_weight"
require_nonnegative_number --ce-weight "$scratch_ce_weight"
if [[ "$scratch_initial_kd_weight" =~ ^0+([.]0*)?$ ]] &&
   [[ "$scratch_initial_ce_weight" =~ ^0+([.]0*)?$ ]]; then
    fail "initial CE and KD weights cannot both be zero"
fi
if [[ "$scratch_kd_weight" =~ ^0+([.]0*)?$ ]] &&
   [[ "$scratch_ce_weight" =~ ^0+([.]0*)?$ ]]; then
    fail "final CE and KD weights cannot both be zero"
fi
require_nonnegative_integer --weight-transition-updates "$scratch_weight_transition_updates"
require_positive_integer --dim "$scratch_dim"
require_positive_integer --heads "$scratch_heads"
require_positive_integer --layers "$scratch_layers"
(( scratch_dim % scratch_heads == 0 )) || fail "--dim must be divisible by --heads"
require_positive_integer --batch-size "$scratch_batch_size"
require_positive_integer --grad-accumulation "$scratch_grad_accumulation"
require_positive_integer --epochs "$scratch_epochs"
require_positive_integer --updates-per-epoch "$scratch_updates_per_epoch"
require_positive_number --segment-duration "$scratch_segment_duration"
require_nonnegative_integer --num-workers "$scratch_num_workers"
require_positive_number --lr "$scratch_lr"
require_positive_integer --valid-samples "$scratch_valid_samples"
require_positive_integer --evaluate-samples "$scratch_evaluate_samples"
require_positive_integer --generate-samples "$scratch_generate_samples"
require_positive_integer --generate-every "$scratch_generate_every"
require_positive_integer --checkpoint-every "$scratch_checkpoint_every"
require_nonnegative_integer --seed "$scratch_seed"

scratch_total_updates=$((scratch_epochs * scratch_updates_per_epoch))
if [[ -z "$scratch_warmup_steps" ]]; then
    scratch_warmup_steps=$(((scratch_total_updates + 19) / 20))
    if (( scratch_warmup_steps >= scratch_total_updates )); then
        scratch_warmup_steps=$((scratch_total_updates - 1))
    fi
fi
require_nonnegative_integer --warmup-steps "$scratch_warmup_steps"
(( scratch_warmup_steps < scratch_total_updates )) ||
    fail "--warmup-steps must be lower than the total number of updates"

scratch_continue_arg=continue_from=null
if [[ -n "$scratch_continue_from" ]]; then
    scratch_continue_arg="continue_from=${scratch_continue_from}"
fi

dora_args=(
    -P audiocraft
    run
    solver=musicgen/musicgen_base_32khz
    model/lm/model_scale=base
    "$scratch_continue_arg"
    conditioner=text2music
    dset=audio/lofi
    "seed=${scratch_seed}"
    "transformer_lm.dim=${scratch_dim}"
    "transformer_lm.num_heads=${scratch_heads}"
    "transformer_lm.num_layers=${scratch_layers}"
    ++transformer_lm.lora.enabled=false
    conditioners.description.t5.finetune=false
    conditioners.description.t5.word_dropout=0
    classifier_free_guidance.training_dropout=0
    dataset.train.merge_text_p=0
    dataset.train.drop_desc_p=0
    dataset.train.drop_other_p=0
    "dataset.num_workers=${scratch_num_workers}"
    "dataset.batch_size=${scratch_batch_size}"
    "dataset.segment_duration=${scratch_segment_duration}"
    dataset.sample_on_weight=false
    dataset.sample_on_duration=false
    dataset.train.permutation_on_files=true
    "dataset.valid.num_samples=${scratch_valid_samples}"
    "dataset.evaluate.num_samples=${scratch_evaluate_samples}"
    "dataset.generate.num_samples=${scratch_generate_samples}"
    "generate.every=${scratch_generate_every}"
    generate.lm.prompted_samples=false
    "optim.epochs=${scratch_epochs}"
    "optim.updates_per_epoch=${scratch_updates_per_epoch}"
    "++optim.grad_accumulation_steps=${scratch_grad_accumulation}"
    optim.optimizer=adamw
    "optim.lr=${scratch_lr}"
    "optim.adam.betas=[0.9,0.95]"
    optim.adam.weight_decay=0.01
    optim.ema.use=false
    schedule.lr_scheduler=cosine
    "schedule.cosine.warmup=${scratch_warmup_steps}"
    schedule.cosine.lr_min_ratio=0.1
    "checkpoint.save_every=${scratch_checkpoint_every}"
    distillation.enabled=true
    "distillation.teacher_checkpoint=${scratch_teacher}"
    "distillation.temperature=${scratch_temperature}"
    "++distillation.initial_kl_weight=${scratch_initial_kd_weight}"
    "++distillation.initial_ce_weight=${scratch_initial_ce_weight}"
    "distillation.kl_weight=${scratch_kd_weight}"
    "distillation.ce_weight=${scratch_ce_weight}"
    "++distillation.weight_schedule_updates=${scratch_weight_transition_updates}"
    "distillation.cfg_branches=${scratch_cfg_branches}"
)

if [[ -n "$scratch_train_data" ]]; then
    dora_args+=("datasource.train=${scratch_train_data}")
fi
if [[ -n "$scratch_valid_data" ]]; then
    dora_args+=(
        "datasource.valid=${scratch_valid_data}"
        "datasource.evaluate=${scratch_valid_data}"
    )
fi

exec dora "${dora_args[@]}"
