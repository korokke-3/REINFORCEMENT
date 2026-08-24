#!/bin/bash
set -e

PYTHON="/home/exhibition-spakona/miniforge3/envs/isaaclab/bin/python"
WORKSPACE="/home/exhibition-spakona/Desktop/REINFORCEMENT"

echo "[$(date)] Starting Right-Leg Single Hopping Training (4096 envs, 800 iterations)..."
cd $WORKSPACE/go2_single_leg

# 本番トレーニング
$PYTHON train.py --num_envs 4096 --max_iterations 800

echo "[$(date)] Training completed! Generating video and trajectory analysis..."
sleep 5

# 動画・解析生成
$PYTHON record_video.py --video_length 300 > logs/record.log 2>&1 || true
$PYTHON analyze_trajectory.py > logs/analyze.log 2>&1 || true

echo "[$(date)] All tasks finished successfully!"
