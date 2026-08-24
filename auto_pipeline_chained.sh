#!/bin/bash

CURRENT_PID=456779
PYTHON="/home/exhibition-spakona/miniforge3/envs/isaaclab/bin/python"
WORKSPACE="/home/exhibition-spakona/Desktop/REINFORCEMENT"

echo "[$(date)] Auto pipeline chained. Monitoring Single-Leg training (PID: $CURRENT_PID)..."

# 1. 1本足けんけんの学習完了待機
while kill -0 $CURRENT_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Single-Leg training completed! Generating video and trajectory..."
sleep 10
cd $WORKSPACE/go2_single_leg
$PYTHON record_video.py --video_length 300 > logs/record.log 2>&1 || true
$PYTHON analyze_trajectory.py > logs/analyze.log 2>&1 || true

# 2. 右側2脚歩行 (右前 + 右後) の学習開始
echo "[$(date)] ==========================================================="
echo "[$(date)] Starting Right-Side Bipedal Training (4096 envs, 800 iters)..."
echo "[$(date)] ==========================================================="
cd $WORKSPACE/go2_right_side
$PYTHON train.py --num_envs 4096 --max_iterations 800 > logs/train.log 2>&1

echo "[$(date)] Right-Side training finished. Generating video and trajectory..."
sleep 10
$PYTHON record_video.py --video_length 300 > logs/record.log 2>&1 || true
$PYTHON analyze_trajectory.py > logs/analyze.log 2>&1 || true

echo "[$(date)] All sequential trainings (Single-Leg & Right-Side) completed successfully!"
sleep 30

systemctl suspend || sudo systemctl suspend || true
