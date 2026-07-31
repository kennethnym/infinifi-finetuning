# Prime Intellect GPU image

This image contains the complete dataset preparation, AudioCraft training, and
generation environment. It uses the official PyTorch 2.1.0 / CUDA 12.1 image
and checks out the exact AudioCraft commit recorded by this repository.

Prime Intellect's custom-image documentation requires `openssh-server` for pod
access. It injects `PUBLIC_KEY` and `SSH_PORT` into the container startup
script. The image installs and configures SSH; the Prime template startup
script handles both variables and starts the server.

Official references:

- [Deploy a custom Docker image](https://docs.primeintellect.ai/tutorials-on-demand-cloud/deploy-custom-docker-image)
- [Provision a GPU pod with a custom template](https://docs.primeintellect.ai/cli-reference/provision-gpu)
- [Attach persistent storage](https://docs.primeintellect.ai/tutorials-storage/use-persistent-storage-with-instances)

## Build and push

Build for the architecture used by Prime Intellect GPU pods:

```bash
docker build --platform linux/amd64 \
  -t <dockerhub-user>/infinifi-finetuning:latest .
docker push <dockerhub-user>/infinifi-finetuning:latest
```

The build downloads the pinned AudioCraft source. It does not include datasets,
Hugging Face models, training outputs, or checkpoints.

## Create the Prime Intellect template

1. Create a custom template using the pushed image name.
2. Set `PUBLIC_KEY` to your complete SSH public key.
3. Set `SSH_PORT` to the template's exposed SSH port (normally `22`).
4. Set the Container Start Script in the template's Advanced section to the
   contents of `prime-entrypoint.sh`:

   ```bash
   #!/bin/bash

   mkdir -p /root/.ssh /var/run/sshd
   chmod 700 /root/.ssh

   if [[ -n "${PUBLIC_KEY:-}" ]]; then
       printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
       chmod 600 /root/.ssh/authorized_keys
   fi

   ssh_port="${SSH_PORT:-22}"
   sed -i '/^#*Port /d' /etc/ssh/sshd_config
   printf 'Port %s\n' "${ssh_port}" >> /etc/ssh/sshd_config

   exec /usr/sbin/sshd -D -e
   ```

5. Select GPU hardware whose NVIDIA driver supports CUDA 12.1.
6. Attach persistent storage if prepared data, Hugging Face caches, Dora runs,
   or checkpoints must survive pod deletion. Prime disks are provider- and
   datacenter-specific, and their mount path is shown after provisioning. Set
   `AUDIOCRAFT_DORA_DIR` and `HF_HOME` to directories on that mount.

The provider supplies the NVIDIA driver and GPU devices at runtime; they must
not be installed in the image.

## Run the pipeline

After connecting as `root`, the repository is available at `/workspace`:

```bash
cd /workspace
python eval/prepare_references.py --dry-run
python eval/prepare_references.py
python eval/generate.py --model facebook/musicgen-small --run-name baseline_musicgen_small --dry-run
python eval/generate.py --model facebook/musicgen-small --run-name baseline_musicgen_small
python prepare.py \
  --candidate-count 20000 \
  --train-count 6000 \
  --valid-count 750 \
  --clap-batch-size 32
bash train_lora.sh \
  --distill \
  --epochs 1 \
  --updates-per-epoch 10 \
  --segment-duration 5 \
  --valid-samples 4 \
  --evaluate-samples 4 \
  --generate-samples 1
DORA_SIGNATURE=aec31258  # Replace with the signature printed by Dora.
python export_adapter.py --signature "$DORA_SIGNATURE" --output-dir /checkpoints/infinifi-lora-kd-r16
python eval/generate.py --model /checkpoints/infinifi-lora-kd-r16 --run-name lora_kd_r16 --dry-run
python eval/generate.py --model /checkpoints/infinifi-lora-kd-r16 --run-name lora_kd_r16
```

Run the baseline generator before fine-tuning. It writes the plain
`facebook/musicgen-small` WAVs and reproducibility manifests under
`/workspace/runs/baseline_musicgen_small`.

`prepare.py` writes its selected dataset and manifests under
`/workspace/audiocraft`.
AudioCraft/Dora writes training runs according to its default grid and cache
configuration. After adapter training, pass the Dora signature to
`export_adapter.py`. The adapter package records the pinned base model and
contains only LoRA tensors; `eval/generate.py` loads the base model, injects
the adapter, and restores its recorded gating mode. The distilled adapter in
the example is active on both conditional and unconditional CFG rows and
writes to the distinct `/workspace/runs/lora_kd_r16` run directory.

### Dataset curation

The complete preparation command streams the pinned `vikhyatk/lofi` revision,
caches 20,000 eligible unique candidates, scores their exact caption/audio
pairs with the pinned LAION-CLAP music checkpoint, and selects exactly 6,000
training plus 750 validation tracks:

```bash
python prepare.py \
  --candidate-count 20000 \
  --train-count 6000 \
  --valid-count 750 \
  --clap-batch-size 32
```

For a small end-to-end smoke test, use a separate cache so its locked
candidate count does not conflict with the full run:

```bash
python prepare.py \
  --candidate-count 128 \
  --train-count 64 \
  --valid-count 16 \
  --clap-batch-size 8 \
  --cache-dir /workspace/curation-smoke
```

The persistent cache defaults to `/workspace/curation`. Candidate MP3s are
written once, their SHA-256 digests are verified on every reuse, and completed
CLAP batches are journaled. Re-running the same command resumes collection or
scoring without downloading or scoring completed valid records. `config.json`
locks the source revision, candidate count, split rules, ignored words, and
CLAP and deduplication identities. If any locked input changes, use a different
`--cache-dir` or deliberately remove the old cache instead of mixing runs. You
may supply the pinned checkpoint locally with `--clap-checkpoint PATH`; its
frozen SHA-256 is still validated.

To manually reject or rewrite captions, create
`/workspace/curation/overrides.jsonl`, or pass another file with
`--overrides PATH`. IDs must be unique and belong to the cached candidate
pool:

```jsonl
{"id":"track-123","action":"drop","reason":"caption/audio mismatch"}
{"id":"track-456","action":"rewrite","caption":"mellow piano lo-fi with soft vinyl crackle","reason":"manually audited"}
```

Drop reasons remain in the audit trail. Rewrites are never generated
automatically: their new caption is rescored, and the effective caption is
used in the final AudioCraft metadata.

Before ranking, preparation removes duplicate audio hashes, exact normalized
captions, and near-identical captions. Near-caption matching uses a
deterministic lexical similarity threshold of `0.90` after punctuation and
whitespace normalization, with efficient one-token-edit candidate matching.
The highest effective CLAP score is retained, with track ID breaking ties.
This catches small wording edits, pluralization, punctuation, and many typos;
it is intentionally not semantic embedding similarity.

The cache contains these reproducibility and audit artifacts:

- `config.json`: locked dataset, split, schema, and CLAP provenance.
- `candidates.jsonl` and `audio/`: source order, captions, split assignments,
  original MP3s, and audio hashes.
- `scores.jsonl`: original/effective captions and their CLAP scores, including
  invalid-audio reasons.
- `selection.jsonl`: every candidate's rank, inclusion or rejection decision,
  override, retained duplicate ID, and audio hash.
- `summary.json`: exact counts, exclusions, deduplication totals, score
  statistics, provenance, and the deterministic `selection.jsonl` digest.

Only selected tracks go through KeyBERT and materialization. Preparation first
builds and validates a staging dataset and exact manifests, then replaces
`audiocraft/dataset/lofi`; collection, scoring, selection, or staging failures
leave the previous final dataset intact. CLAP is a caption/audio filtering
signal, not a complete measure of musical quality, production quality, or
dataset suitability.

To compare periodic epoch checkpoints, export each one to a separate package.
`--checkpoint` accepts a positive epoch number; omitting it, or passing
`latest`, exports `checkpoint.th`:

```bash
python export_checkpoint.py \
  --signature "$DORA_SIGNATURE" \
  --checkpoint 1 \
  --output-dir /checkpoints/infinifi-epoch1
```

Generation uses classifier-free guidance coefficient 3.0 by default. Give
every alternate coefficient a distinct run name because the value is locked
into the run configuration:

```bash
python eval/generate.py \
  --model /checkpoints/infinifi-epoch1 \
  --run-name finetuned_infinifi_epoch1_cfg4 \
  --cfg-coef 4
```

### LoRA adapter tuning

`train_lora.sh` freezes MusicGen, T5, and EnCodec and trains gated LoRA
projections in every self-attention, cross-attention, and feed-forward block.
Without `--distill`, the adapter is disabled per sample when every conditioning
mask is empty, so the CFG null branch remains the exact pretrained model. The
ordinary LoRA defaults run 1,500 updates with rank 8, alpha 8, adapter dropout
0.05, and learning rate `1e-4`:

```bash
bash train_lora.sh --rank 8
bash train_lora.sh --rank 16
```

Run the rank experiments with otherwise identical arguments. Alpha defaults
to the selected rank, and both runs use seed 2036. Classifier-free, T5 word,
and metadata dropout are disabled so rank is the controlled variable.
Full-model EMA and FSDP are not supported by this adapter path.

For teacher-student distillation, add `--distill`. The solver loads
`facebook/musicgen-large` as a frozen teacher and feeds the teacher and student
the same prompt and EnCodec prefix. It minimizes a temperature-scaled,
codebook-masked `KL(teacher || student)` loss mixed with the existing hard-token
cross entropy. Only the Small model's LoRA tensors receive optimizer updates:

```bash
bash train_lora.sh --distill
```

Distillation defaults change to rank 16, batch size 1, and learning rate
`3e-5`; the loss defaults to `0.75 * KD + 0.25 * CE` at temperature 2.
Conditional and unconditional branches are explicitly distilled, so the
adapter is active on both CFG rows whenever it is loaded. Use
`--conditional-only` only for an intentional ablation. Loading Large alongside
the student raises accelerator memory use substantially; reduce segment
duration before changing the loss if the run is memory constrained.

Run `bash train_lora.sh --help` for teacher, temperature, loss-weight, rank,
batch, and scheduling controls.

Export each rank or periodic epoch to its own immutable package:

```bash
python export_adapter.py \
  --signature "$DORA_SIGNATURE" \
  --checkpoint 1 \
  --output-dir /checkpoints/infinifi-lora-r8-epoch1
```

The package contains `adapter.json` and `adapter_state.bin`. Its metadata
records whether activation follows the condition mask or adapter loading.
Condition-gated adapters deliberately cannot be merged into the base weights
because merging would also alter null conditioning.

For inference-only scaling experiments, keep the exported package immutable
and pass `--adapter-scale` to the evaluator. The scale defaults to `1.0`, is
recorded in the locked run configuration, and multiplies every LoRA projection:

```bash
python eval/generate.py \
  --model /checkpoints/infinifi-lora-r8-epoch1 \
  --run-name lora_r8_epoch1_cfg3_scale033 \
  --cfg-coef 3 \
  --adapter-scale 0.3333333333
```

Use scale `0` for a base-model equivalence check. Non-default values are only
accepted for LoRA adapter packages.

### From-scratch compact student distillation

`train_scratch.sh` randomly initializes only the music-token LM. Its default
student is a MusicGen-compatible 10-layer, width-640 transformer with 10 heads
(about 77M trainable parameters), four EnCodec codebooks, and cardinality 2048.
The pretrained EnCodec model and T5 backbone are frozen; the student's small
T5-to-LM projection remains trainable because its output width is student-specific.
MusicGen-Large is loaded frozen and is never stored in the student checkpoint.

The default first stage is 20,000 optimizer updates over random 10-second crops.
Batch size 1 with eight-way gradient accumulation gives an effective per-GPU
batch of eight. `updates-per-epoch` continues to mean optimizer updates, not
microbatches. Both CFG branches are distilled explicitly, and loss weights
move linearly from `0.5 KD + 0.5 CE` to `0.75 KD + 0.25 CE` during the first
10,000 updates:

```bash
bash train_scratch.sh
```

First validate the path by overfitting a fixed 100–500-record manifest. The
manifest rows can refer to the existing prepared audio and metadata:

```bash
mkdir -p /workspace/egs/overfit
head -n 250 audiocraft/egs/train/data.jsonl > /workspace/egs/overfit/data.jsonl
bash train_scratch.sh \
  --train-data /workspace/egs/overfit \
  --epochs 2 \
  --updates-per-epoch 100 \
  --segment-duration 5 \
  --weight-transition-updates 100
```

For the 20-second continuation, start a new Dora experiment from the stage-one
student. Keep the architecture unchanged and begin directly at the final loss
mixture:

```bash
bash train_scratch.sh \
  --continue-from "//sig/$STAGE_ONE_SIGNATURE" \
  --segment-duration 20 \
  --epochs 30 \
  --updates-per-epoch 1000 \
  --initial-kd-weight 0.75 \
  --initial-ce-weight 0.25 \
  --weight-transition-updates 0
```

Export it with `export_checkpoint.py`, then use the normal local-package
generation and scoring commands. Run `bash train_scratch.sh --help` for model,
loss, accumulation, crop, resume, and manifest controls. The Large teacher's
CC-BY-NC weight license still requires legal review for commercial use of the
distilled student.

### Full-model fine-tuning (legacy)

`train.sh` fine-tunes the pretrained `facebook/musicgen-small` model on the
prepared lo-fi dataset. It does not perform teacher-student distillation. Its
single-GPU defaults run 5,000 optimizer updates as 10 epochs of 500 updates,
using 20-second segments and a batch size of two:

```bash
bash train.sh
```

Run the default training configuration only after preparing a meaningfully
sized dataset. The short training invocation in the pipeline example is an
end-to-end smoke test; it is not intended to produce a quality-ready
checkpoint.

Pass command-line options to change the training budget and memory-sensitive
settings without editing the script:

| Option | Default |
| --- | ---: |
| `--batch-size` | `2` |
| `--epochs` | `10` |
| `--updates-per-epoch` | `500` |
| `--segment-duration` | `20` |
| `--num-workers` | `4` |
| `--lr` | `1e-5` |
| `--warmup-steps` | 5% of total updates (`250`) |
| file sampling | deterministic permutation |
| `--valid-samples` | `128` |
| `--evaluate-samples` | `128` |
| `--generate-samples` | `4` |
| `--generate-every` | `5` |
| `--checkpoint-every` | `5` |
| `--word-dropout` | AudioCraft configuration |
| `--cfg-dropout` | AudioCraft configuration |
| `--merge-text-p` | AudioCraft configuration |
| `--drop-desc-p` | AudioCraft configuration |
| `--drop-other-p` | AudioCraft configuration |

Run `bash train.sh --help` for the complete command reference. Set
`--batch-size 1` if training runs out of GPU memory. On distributed runs,
the default file permutation derives the number of training samples from the
batch size multiplied by updates per epoch and world size. Set updates per
epoch to the selected track count divided by that global batch size when one
epoch should be one complete pass.

For a gentler, text-focused fine-tune that preserves more of the pretrained
model while saving every epoch:

```bash
bash train.sh \
  --lr 2e-6 \
  --epochs 3 \
  --updates-per-epoch 500 \
  --warmup-steps 100 \
  --word-dropout 0.1 \
  --cfg-dropout 0.1 \
  --merge-text-p 0 \
  --drop-desc-p 0 \
  --drop-other-p 0 \
  --generate-every 1 \
  --checkpoint-every 1
```

Reference preparation downloads two 500-track corpora under
`/workspace/references`. Put `--output-root` on attached persistent storage if
the corpora must survive pod deletion. The software is MIT-licensed, but the
downloaded reference audio retains its source Creative Commons licenses and
includes non-commercial tracks. See `eval/README.md` for the exact source
revisions, selection rules, and attribution files.

The prepared `musicgen-large-v1` corpus contains all twenty paired
`dataset_eval` references required by KLD. Score both generated corpora after
reference preparation and generation complete:

```bash
python eval/score.py --run-name baseline_musicgen_small --dry-run
python eval/score.py --run-name baseline_musicgen_small
python eval/score.py --run-name finetuned_infinifi --dry-run
python eval/score.py --run-name finetuned_infinifi
```

The scorer writes aggregate metrics and per-clip scores beside each generated
run. See `eval/README.md` for metric definitions, checkpoint requirements, and
output formats.

The image defaults `AUDIOCRAFT_DORA_DIR` to `/workspace/.cache/dora`; override
it with a persistent-disk path for real training.

For private Hugging Face resources, inject `HF_TOKEN` through the template
instead of adding credentials to the image. The image enables the accelerated
Hugging Face transfer client and stores its default cache at
`/workspace/.cache/huggingface`.

## Local smoke test

An NVIDIA Container Toolkit installation is required for the GPU check:

```bash
docker run --rm --gpus all \
  <dockerhub-user>/infinifi-finetuning:latest \
  python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.is_available())"
```

To test SSH, mount and run the same startup script used by the Prime template:

```bash
docker run --rm --gpus all --ipc=host --shm-size=8g \
  -p 2222:22 \
  -e SSH_PORT=22 \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  -v "$(pwd)/prime-entrypoint.sh:/prime-entrypoint.sh:ro" \
  --entrypoint /bin/bash \
  <dockerhub-user>/infinifi-finetuning:latest \
  /prime-entrypoint.sh
```
