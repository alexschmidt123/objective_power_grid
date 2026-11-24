#!/bin/bash
# Wait for a specific process instance, then execute an approved shell entrypoint.
set -euo pipefail
WAIT_PID=${1:?Usage: queue_after_pid.sh PID command arguments...}
shift
[[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { echo "Invalid PID" >&2; exit 2; }
process_identity() {
    [[ -r "/proc/$WAIT_PID/stat" ]] || return 1
    awk '{if ($3 == "Z") exit 1; print $22}' "/proc/$WAIT_PID/stat"
}
if START_ID=$(process_identity); then
    echo "Waiting for PID $WAIT_PID (process start $START_ID)"
    while CURRENT_ID=$(process_identity) && [[ "$CURRENT_ID" == "$START_ID" ]]; do
        sleep 15
    done
fi
echo "Preceding process exited; starting queued command at $(date -Is)"
exec "$@"
