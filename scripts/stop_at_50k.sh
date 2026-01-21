#!/bin/bash
# Monitor training and stop at 50k iterations

TARGET_ITER=50000
LOG_FILE="/home/naoto/workspace/PolarFree/logs/psd_stage2_train.log"
MODEL_DIR="/home/naoto/workspace/PolarFree/experiments/psd_stage2/models"

echo "Monitoring Stage 2 training to stop at ${TARGET_ITER} iterations..."
echo "Checking every 5 minutes..."

while true; do
    # Get latest iteration from log
    LATEST_LINE=$(grep -oP 'iter:\s*\K[0-9,]+' "$LOG_FILE" | tail -1 | tr -d ',')

    if [ -n "$LATEST_LINE" ]; then
        echo "$(date): Current iteration: $LATEST_LINE"

        if [ "$LATEST_LINE" -ge "$TARGET_ITER" ]; then
            echo "Reached ${TARGET_ITER} iterations!"

            # Wait for checkpoint to be saved
            sleep 60

            # Check if checkpoint exists
            if [ -f "${MODEL_DIR}/net_g_${TARGET_ITER}.pth" ]; then
                echo "Checkpoint saved. Stopping training..."
                pkill -f "train.py -opt options/train/psd_stage2.yml"
                echo "Training stopped at iteration ${TARGET_ITER}"
                exit 0
            else
                echo "Waiting for checkpoint to be saved..."
            fi
        fi
    fi

    sleep 300  # Check every 5 minutes
done
