#!/bin/bash
# Open a shell (or run a command) inside the danns container, locally or via SSH.
#
# Usage:
#   ./danns-exec.sh                                  # local container shell
#   ./danns-exec.sh node1                            # shell on remote node1
#   ./danns-exec.sh node1 "python3 scripts/..."      # run command on node1
#
# REMOTE_ARTIFACT_ROOT must point at the artifact root on the remote node
# (where docker-compose.yml lives). It is required for remote invocations
# only and may be exported in your shell or set inline:
#   REMOTE_ARTIFACT_ROOT=/srv/sparsepath_artifact ./danns-exec.sh node1 ...

NODE="${1:-}"
COMMAND="${2:-bash}"

EXEC_CMD="docker compose exec -e HOST_UID=\$(id -u) -e HOST_GID=\$(id -g) -e USERNAME=\$(whoami) danns $COMMAND"

if [ -z "$NODE" ]; then
    # Local
    eval "$EXEC_CMD"
else
    # Remote via SSH
    : "${REMOTE_ARTIFACT_ROOT:?REMOTE_ARTIFACT_ROOT must be set for remote invocations}"
    ssh "$NODE" "cd '$REMOTE_ARTIFACT_ROOT' && $EXEC_CMD"
fi
