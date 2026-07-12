#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.ssh /var/run/sshd
chmod 700 /root/.ssh

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

ssh_port="${SSH_PORT:-22}"
if [[ ! "${ssh_port}" =~ ^[0-9]+$ ]] || ((ssh_port < 1 || ssh_port > 65535)); then
    echo "SSH_PORT must be an integer between 1 and 65535." >&2
    exit 1
fi

sed -i '/^#*Port /d' /etc/ssh/sshd_config
printf 'Port %s\n' "${ssh_port}" >> /etc/ssh/sshd_config

# Generate unique host keys when each pod starts instead of baking shared keys
# into the image.
ssh-keygen -A

echo "Starting SSH server on port ${ssh_port}"
exec /usr/sbin/sshd -D -e
