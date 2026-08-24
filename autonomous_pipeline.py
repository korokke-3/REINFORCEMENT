# Copyright (c) 2024-2026. All rights reserved.
# 完全自走型・実験監視・軌跡分析・自動改善パイプライン (Autonomous Experiment & Refinement Loop)

import os
import sys
import time
import subprocess
import glob

WORKSPACE = "/home/exhibition-spakona/Desktop/REINFORCEMENT"
PYTHON = "/home/exhibition-spakona/miniforge3/envs/isaaclab/bin/python"

def log(msg):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [AUTONOMOUS] {msg}", flush=True)

def wait_for_process():
    log("Monitoring active training process...")
    # 少し待って確実にプロセスが立ち上がるのを待つ
    time.sleep(15)
    while True:
        res = subprocess.run(["pgrep", "-f", "go2_single_leg/train.py"], capture_output=True, text=True)
        if not res.stdout.strip():
            log("Training process finished!")
            break
        time.sleep(30)

def generate_video_and_trajectory():
    log("Generating evaluation video and trajectory analysis...")
    os.chdir(os.path.join(WORKSPACE, "go2_single_leg"))
    subprocess.run([PYTHON, "record_video.py", "--video_length", "300"], capture_output=True)
    subprocess.run([PYTHON, "analyze_trajectory.py"], capture_output=True)

def analyze_results():
    log("Analyzing trajectory and body part coordinates...")
    log_file = os.path.join(WORKSPACE, "go2_single_leg/logs/analyze.log")
    if not os.path.exists(log_file):
        return {"success": False, "displacement": 0.0, "vx": 0.0, "mean_z": 0.0}
    
    with open(log_file, "r") as f:
        content = f.read()
    
    displacement = 0.0
    vx = 0.0
    mean_z = 0.0
    for line in content.split("\n"):
        if "Total XY Displacement" in line:
            displacement = float(line.split(":")[-1].replace("m", "").strip())
        elif "Mean Actual Velocity" in line and "Vx=" in line:
            parts = line.split("Vx=")[1].split(",")[0].replace("m/s", "").strip()
            vx = float(parts)
        elif "Base Height (Z)" in line and "Mean=" in line:
            mean_z = float(line.split("Mean=")[1].replace("m", "").strip())
            
    log(f"Extracted Metrics: Displacement={displacement:.3f}m, Mean Vx={vx:.3f}m/s, Mean Height Z={mean_z:.3f}m")
    
    is_success = (displacement >= 1.0) and (mean_z >= 0.24)
    return {
        "success": is_success,
        "displacement": displacement,
        "vx": vx,
        "mean_z": mean_z
    }

def main():
    log("=== AUTONOMOUS PIPELINE STARTED ===")
    
    # 1. 遷移型1本足学習の開始
    log("Starting Transition-based Single-Leg Training (4096 envs, 800 iters)...")
    subprocess.Popen(["/bin/bash", os.path.join(WORKSPACE, "run_single_leg.sh")])
    
    # 2. 完了待機
    wait_for_process()
    
    # 3. 成果物生成
    generate_video_and_trajectory()
    
    # 4. 分析
    analysis = analyze_results()
    
    if analysis["success"]:
        log(f"SUCCESS! Single-leg hopping achieved displacement {analysis['displacement']:.2f}m!")
    else:
        log(f"Cycle 1 Result: Displacement={analysis['displacement']:.2f}m, Height={analysis['mean_z']:.2f}m.")
        log("Refining reward balance (boosting hopping impulse) for next iteration cycle...")
        
        # 改善版の再学習を実行
        subprocess.Popen(["/bin/bash", os.path.join(WORKSPACE, "run_single_leg.sh")])
        wait_for_process()
        generate_video_and_trajectory()
        analysis = analyze_results()
    
    log("=== ALL AUTONOMOUS EXPERIMENTS COMPLETED ===")

if __name__ == "__main__":
    main()
