#!/bin/bash

INITIAL_PID=361081
PYTHON="/home/exhibition-spakona/miniforge3/envs/isaaclab/bin/python"
WORKSPACE="/home/exhibition-spakona/Desktop/REINFORCEMENT"

echo "[$(date)] Pipeline started. Waiting for current initial single-leg run (PID: $INITIAL_PID) to finish..."

# 1. 現在走っている初代1本足学習の完了待機
while kill -0 $INITIAL_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Initial single-leg run finished. Generating initial video..."
sleep 10
cd $WORKSPACE/go2_single_leg
$PYTHON record_video.py --video_length 300 > logs/initial_record.log 2>&1 || true
$PYTHON analyze_trajectory.py > logs/initial_analyze.log 2>&1 || true

# 2. 修正版 2本足歩行 (直立スポーン & 前進ゲーティング) の学習
echo "[$(date)] ================================================"
echo "[$(date)] Starting Fixed Bipedal Training (800 iterations)..."
echo "[$(date)] ================================================"
cd $WORKSPACE/go2_bipedal
$PYTHON train.py --num_envs 4096 --max_iterations 800 > logs/train_fixed.log 2>&1

echo "[$(date)] Fixed Bipedal training finished. Generating video and trajectory..."
sleep 10
$PYTHON record_video.py --video_length 300 > logs/record.log 2>&1 || true
$PYTHON analyze_trajectory.py > logs/analyze.log 2>&1 || true

# 3. 修正版 1本足ダイナミックけんけん (1脚スポーン & 前進ゲーティング) の学習
echo "[$(date)] ==================================================="
echo "[$(date)] Starting Fixed Single-Leg Training (800 iterations)..."
echo "[$(date)] ==================================================="
cd $WORKSPACE/go2_single_leg
$PYTHON train.py --num_envs 4096 --max_iterations 800 > logs/train_fixed.log 2>&1

echo "[$(date)] Fixed Single-Leg training finished. Generating video and trajectory..."
sleep 10
$PYTHON record_video.py --video_length 300 > logs/record.log 2>&1 || true
$PYTHON analyze_trajectory.py > logs/analyze.log 2>&1 || true

echo "[$(date)] All tasks and experiments completed successfully! Suspending system..."
sleep 30

systemctl suspend || sudo systemctl suspend || true
