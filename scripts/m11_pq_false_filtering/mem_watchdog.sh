#!/bin/bash
# Shared-node protection: kill run_m11.py if host available memory drops below
# THRESH_MB, so a heavy single-node run (e.g. LAION 512-d, ~248GB) can never OOM
# a node that hosts other sessions. Worst case only OUR run dies.
# Usage: mem_watchdog.sh [THRESH_MB]   (default 40000 = 40GB)
THRESH=${1:-40000}
echo "WATCHDOG start thresh=${THRESH}MB $(date -u)"
while pgrep -f "[r]un_m11" >/dev/null; do
  a=$(free -m | awk 'NR==2{print $7}')   # available MB (host, no cgroup limit)
  if [ "$a" -lt "$THRESH" ]; then
    pkill -9 -f run_m11.py
    echo "WATCHDOG_KILL avail=${a}MB < ${THRESH}MB -> killed run_m11 $(date -u)"
    exit 1
  fi
  sleep 5
done
echo "WATCHDOG_EXIT run_m11 finished on its own $(date -u)"
