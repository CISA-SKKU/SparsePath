#!/bin/bash
# DANNS entrypoint — runtime UID/GID matching
# Usage: docker run -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) danns
#
# If HOST_UID/HOST_GID are set, create/update a user with matching IDs.
# This lets any team member's files have correct ownership on NFS.

set -e

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
USERNAME="${USERNAME:-dev}"

# Create group if it doesn't exist
if ! getent group "$HOST_GID" > /dev/null 2>&1; then
    groupadd -g "$HOST_GID" hostgroup
fi
GROUP_NAME=$(getent group "$HOST_GID" | cut -d: -f1)

# Create or update user
if ! id "$USERNAME" > /dev/null 2>&1; then
    useradd -u "$HOST_UID" -g "$HOST_GID" -m -s /bin/bash "$USERNAME"
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# Setup conda for this user
USERHOME=$(eval echo "~$USERNAME")
if [ ! -f "$USERHOME/.bashrc" ] || ! grep -q 'conda activate danns' "$USERHOME/.bashrc"; then
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> "$USERHOME/.bashrc"
    echo 'conda activate danns' >> "$USERHOME/.bashrc"
    echo 'export PATH="/workspace/build/apps:/workspace/build/apps/utils:$PATH"' >> "$USERHOME/.bashrc"
fi
chown -R "$HOST_UID:$HOST_GID" "$USERHOME" 2>/dev/null || true

# Optional host RDMA device rename. Worker containers run privileged with
# host networking, so this keeps rdma-core persistent names aligned with the
# mlx5_0/mlx5_1 names expected by nodes.yaml after container recreation.
if [ -n "${RDMA_RENAME_PAIRS:-}" ] && command -v rdma >/dev/null 2>&1; then
    IFS=',' read -ra _rdma_pairs <<< "$RDMA_RENAME_PAIRS"
    for _pair in "${_rdma_pairs[@]}"; do
        _from="${_pair%%:*}"
        _to="${_pair##*:}"
        if [ -z "$_from" ] || [ -z "$_to" ] || [ "$_from" = "$_to" ]; then
            continue
        fi
        if rdma dev show "$_to" >/dev/null 2>&1; then
            continue
        fi
        if rdma dev show "$_from" >/dev/null 2>&1; then
            rdma dev set "$_from" name "$_to" || true
        fi
    done
fi

# Run command as the matched user
source /opt/conda/etc/profile.d/conda.sh
conda activate danns

if [ "$HOST_UID" = "0" ]; then
    exec "$@"
else
    exec gosu "$USERNAME" "$@"
fi
