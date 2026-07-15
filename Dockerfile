FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG AUDIOCRAFT_COMMIT=adf0b04a4452f171970028fcf80f101dd5e26e19

ENV DEBIAN_FRONTEND=noninteractive \
    AUDIOCRAFT_DORA_DIR=/workspace/.cache/dora \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install SSH server and pipeline dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libsndfile1-dev \
        openssh-server \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd \
    && printf '\nPasswordAuthentication no\nPubkeyAuthentication yes\n' >> /etc/ssh/sshd_config \
    && ssh-keygen -A

# Create SSH directory for root
RUN mkdir -p /root/.ssh /checkpoints/infinifi \
    && chmod 700 /root/.ssh

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba install -y -p /opt/conda -c conda-forge --strict-channel-priority \
        av=11.0.0 \
        "ffmpeg>=5,<7" \
    && micromamba clean --all --yes \
    && rm /usr/local/bin/micromamba

WORKDIR /workspace

COPY audiocraft-requirements.txt pipeline-requirements.txt ./

RUN sed '/^nvidia-/d; /^pip==/d; /^setuptools==/d' \
        audiocraft-requirements.txt > /tmp/audiocraft-requirements.txt \
    && python -m pip install --no-cache-dir -r /tmp/audiocraft-requirements.txt \
    && python -m pip install --no-cache-dir -r pipeline-requirements.txt

RUN git clone https://github.com/facebookresearch/audiocraft.git /workspace/audiocraft \
    && git -C /workspace/audiocraft checkout --detach "${AUDIOCRAFT_COMMIT}" \
    && python -m pip install --no-cache-dir --no-deps -e /workspace/audiocraft \
    && rm -rf /workspace/audiocraft/.git

COPY prepare.py export_checkpoint.py train.sh ./
COPY eval ./eval

RUN chmod +x /workspace/train.sh \
    && python -c "import audiocraft, datasets, keybert, torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"

EXPOSE 22
