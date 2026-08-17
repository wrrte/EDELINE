#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

QUEUE_FILE="job_queue.txt"
LOCK_FILE="job_queue.lock"

touch "$QUEUE_FILE"

echo "Worker started on GPU 6. Waiting for jobs in $QUEUE_FILE..."

while true; do
    job=""
    
    # Critical section to safely pop a job
    exec 200>"$LOCK_FILE"
    flock -x 200
    
    # Remove empty lines
    sed -i '/^[[:space:]]*$/d' "$QUEUE_FILE"
    
    if [ -s "$QUEUE_FILE" ]; then
        # Read the first line
        job=$(head -n 1 "$QUEUE_FILE")
        # Delete the first line
        sed -i '1d' "$QUEUE_FILE"
    fi
    
    flock -u 200
    exec 200>&-

    if [ -n "$job" ]; then
        echo "[GPU 6] Found job: $job"
        eval "$job"
        echo "[GPU 6] Job finished."
    else
        # Sleep for 10 seconds before polling again
        sleep 10
    fi
done