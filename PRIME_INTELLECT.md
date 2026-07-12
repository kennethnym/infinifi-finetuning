# Prime Intellect GPU image

This image contains the complete dataset preparation, AudioCraft training, and
generation environment. It uses the official PyTorch 2.1.0 / CUDA 12.1 image
and checks out the exact AudioCraft commit recorded by this repository.

Prime Intellect's custom-image documentation requires `openssh-server` for pod
access. It injects `PUBLIC_KEY` and `SSH_PORT` into the container startup
script. This image handles both variables in its default command and generates
unique SSH host keys each time a pod starts.

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
4. Leave the container start script empty so the image's default command runs.
   If the template requires a script, use:

   ```bash
   #!/bin/bash
   exec /usr/local/bin/prime-entrypoint
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
BATCH_COUNT=1 python prepare.py
bash train.sh
python generate.py
```

Increase `BATCH_COUNT` after the one-batch preparation test succeeds.
`prepare.py` writes its dataset and manifests under `/workspace/audiocraft`.
AudioCraft/Dora writes training runs according to its default grid and cache
configuration. Before generation, update the hard-coded Dora experiment
signature in `generate.py` to the signature printed by the new training run.
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

Because the image has a `CMD`, supplying `python ...` as above replaces the SSH
server for one-off checks. To test SSH, run the default command:

```bash
docker run --rm --gpus all --ipc=host --shm-size=8g \
  -p 2222:22 \
  -e SSH_PORT=22 \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  <dockerhub-user>/infinifi-finetuning:latest
```
