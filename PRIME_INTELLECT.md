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
BATCH_COUNT=1 python prepare.py
bash train.sh \
  --epochs 1 \
  --updates-per-epoch 10 \
  --segment-duration 5 \
  --valid-samples 4 \
  --evaluate-samples 4 \
  --generate-samples 1
DORA_SIGNATURE=aec31258  # Replace with the signature printed by Dora.
python export_checkpoint.py --signature "$DORA_SIGNATURE" --output-dir /checkpoints/infinifi
python eval/generate.py --model /checkpoints/infinifi --run-name finetuned_infinifi --dry-run
python eval/generate.py --model /checkpoints/infinifi --run-name finetuned_infinifi
```

Run the baseline generator before fine-tuning. It writes the plain
`facebook/musicgen-small` WAVs and reproducibility manifests under
`/workspace/runs/baseline_musicgen_small`.

Increase `BATCH_COUNT` after the one-batch preparation test succeeds.
`prepare.py` writes its dataset and manifests under `/workspace/audiocraft`.
AudioCraft/Dora writes training runs according to its default grid and cache
configuration. After training, pass the Dora signature to
`export_checkpoint.py`. Exporting only packages the LM and pinned
`facebook/encodec_32khz` compression model; `eval/generate.py` performs audio
generation for both pretrained model IDs and exported package directories.
The fine-tuned example writes to the distinct
`/workspace/runs/finetuned_infinifi` run directory.

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

### Fine-tuning

`train.sh` fine-tunes the pretrained `facebook/musicgen-small` model on the
prepared lo-fi dataset. It does not perform teacher-student distillation. Its
single-GPU defaults run 5,000 optimizer updates as 10 epochs of 500 updates,
using 20-second segments and a batch size of two:

```bash
bash train.sh
```

Run the default training configuration only after increasing `BATCH_COUNT` and
preparing a meaningfully sized dataset. The short invocation in the pipeline
example is an end-to-end smoke test; it is not intended to produce a
quality-ready checkpoint.

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
| `--train-samples` | batch size × updates per epoch (`1000`) |
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
explicitly set `--train-samples` to at least the batch size multiplied by
updates per epoch and world size.

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
