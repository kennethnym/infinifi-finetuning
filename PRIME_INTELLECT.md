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
python eval/generate.py --model facebook/musicgen-small --run-name baseline_musicgen_small --dry-run
python eval/generate.py --model facebook/musicgen-small --run-name baseline_musicgen_small
BATCH_COUNT=1 python prepare.py
bash train.sh
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
