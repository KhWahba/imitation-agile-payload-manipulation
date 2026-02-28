#!/usr/bin/env bash
set -euo pipefail
cd /home/khaledwahba94/imitation-agile-payload-manipulation
PID=""
while [[ -z "$PID" ]]; do
  PID=$(pgrep -nf "python3 .*scripts/train_dagger_payload.py|python .*scripts/train_dagger_payload.py" || true)
  sleep 1
done
{
  echo "Monitoring PID=$PID"
  while kill -0 "$PID" 2>/dev/null; do
    printf "%s " "$(date '+%H:%M:%S')"
    awk '/VmRSS:|VmHWM:/{printf "%s%s ", $1, $2} END{print ""}' /proc/$PID/status
    sleep 2
  done
} | tee scripts/ram_watch.log
