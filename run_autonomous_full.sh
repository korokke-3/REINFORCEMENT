#!/bin/bash
set -e

WORKSPACE="/home/exhibition-spakona/Desktop/REINFORCEMENT"
PYTHON="/home/exhibition-spakona/miniforge3/envs/isaaclab/bin/python"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTONOMOUS] $1" | tee -a "$WORKSPACE/autonomous_full.log"
}

log "=== FULL AUTONOMOUS CONTINUOUS LEARNING STARTED ==="

# ----------------------------------------------------
# Cycle 1: 遷移型 1本足けんけん本番学習 (800 iters)
# ----------------------------------------------------
log "Cycle 1: Starting Transition-based Single-Leg Training (4096 envs, 800 iters)..."
cd "$WORKSPACE/go2_single_leg"
$PYTHON train.py --num_envs 4096 --max_iterations 800 | tee "$WORKSPACE/go2_single_leg/logs/train_cycle1.log"

log "Cycle 1: Training finished. Generating video and trajectory..."
$PYTHON record_video.py --video_length 300 > "$WORKSPACE/go2_single_leg/logs/record.log" 2>&1
$PYTHON analyze_trajectory.py > "$WORKSPACE/go2_single_leg/logs/analyze.log" 2>&1

# ----------------------------------------------------
# 分析と判定
# ----------------------------------------------------
log "Analyzing trajectory and coordinates..."
cat "$WORKSPACE/go2_single_leg/logs/analyze.log" | grep -E "Displacement|Velocity|Base Height" | tee -a "$WORKSPACE/autonomous_full.log"

log "=== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ==="
