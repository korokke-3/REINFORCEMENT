"""
強化学習 (RSL-RL PPO) の学習ログから収束状況・平均報酬・エピソード長・完走率を解析・表示するスクリプト
"""
from __future__ import annotations

import argparse
import glob
import os
import re


def parse_log(log_path: str):
    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found: {log_path}")
        return

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    iter_blocks = text.split("Learning iteration ")
    if len(iter_blocks) <= 1:
        print(f"[WARN] No learning iteration records found in {log_path}")
        return

    iterations = []
    rewards = []
    ep_lengths = []
    val_losses = []
    timeouts = []

    for block in iter_blocks[1:]:
        lines = block.split("\n")
        it_match = re.match(r"(\d+)/\d+", lines[0])
        if not it_match:
            continue
        it_num = int(it_match.group(1))

        rew_match = re.search(r"Mean reward:\s+([\d\.\-]+)", block)
        len_match = re.search(r"Mean episode length:\s+([\d\.\-]+)", block)
        val_match = re.search(r"Mean value loss:\s+([\d\.\-]+)", block)
        time_match = re.search(r"Episode_Termination/time_out:\s+([\d\.\-]+)", block)

        if rew_match and len_match:
            iterations.append(it_num)
            rewards.append(float(rew_match.group(1)))
            ep_lengths.append(float(len_match.group(1)))
            val_losses.append(float(val_match.group(1)) if val_match else 0.0)
            timeouts.append(float(time_match.group(1)) if time_match else 0.0)

    if not iterations:
        print(f"[WARN] No valid iteration metrics parsed from {log_path}")
        return

    print(f"\n================================================================================")
    print(f"=== TRAINING CONVERGENCE SUMMARY: {os.path.basename(log_path)} ===")
    print(f"Total Logged Iterations: {len(iterations)} (from {iterations[0]} to {iterations[-1]})")
    print(f"Peak Mean Reward        : {max(rewards):.2f}")
    print(f"Max Episode Length      : {max(ep_lengths):.2f} steps ({max(ep_lengths)*0.02:.2f} s)")
    print(f"Peak TimeOut Rate       : {max(timeouts)*100:.1f}%")
    print(f"================================================================================")

    step_count = min(5, len(iterations))
    sample_indices = [int(i * (len(iterations) - 1) / (step_count - 1)) for i in range(step_count)]

    print(f"{'Iteration':<12} | {'Mean Reward':<14} | {'Ep Length (steps)':<18} | {'TimeOut %':<12} | {'Value Loss':<12}")
    print("-" * 80)
    for idx in sample_indices:
        it = iterations[idx]
        r = rewards[idx]
        l = ep_lengths[idx]
        t = timeouts[idx] * 100
        v = val_losses[idx]
        print(f"{it:<12} | {r:<14.2f} | {l:<18.2f} | {t:<11.1f}% | {v:<12.1f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze RSL-RL PPO Training Log Convergence")
    parser.add_argument("--log_file", type=str, default=None, help="Path to training .log file")
    args = parser.parse_args()

    if args.log_file:
        parse_log(args.log_file)
    else:
        # Search for any log files in logs directory
        log_files = glob.glob("**/train*.log", recursive=True) + glob.glob("**/*.log", recursive=True)
        if log_files:
            for lf in log_files[:3]:
                parse_log(lf)
        else:
            print("[INFO] No log files specified. Pass --log_file <path/to/log> to analyze a training run.")


if __name__ == "__main__":
    main()
