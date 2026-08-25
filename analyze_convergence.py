import re
import numpy as np

log_file = "/home/exhibition-spakona/.gemini/antigravity-cli/brain/5911a837-9092-4123-82e7-95f98b9a8527/.system_generated/tasks/task-658.log"

iterations = []
rewards = []
ep_lengths = []
val_losses = []
timeouts = []

with open(log_file, 'r') as f:
    text = f.read()

iter_blocks = text.split("Learning iteration ")

for block in iter_blocks[1:]:
    lines = block.split('\n')
    it_match = re.match(r'(\d+)/\d+', lines[0])
    if not it_match: continue
    it_num = int(it_match.group(1))
    
    rew_match = re.search(r'Mean reward:\s+([\d\.\-]+)', block)
    len_match = re.search(r'Mean episode length:\s+([\d\.\-]+)', block)
    val_match = re.search(r'Mean value loss:\s+([\d\.\-]+)', block)
    time_match = re.search(r'Episode_Termination/time_out:\s+([\d\.\-]+)', block)
    
    if rew_match and len_match:
        iterations.append(it_num)
        rewards.append(float(rew_match.group(1)))
        ep_lengths.append(float(len_match.group(1)))
        val_losses.append(float(val_match.group(1)) if val_match else 0.0)
        timeouts.append(float(time_match.group(1)) if time_match else 0.0)

print(f"Total Logged Iterations: {len(iterations)} (from {iterations[0]} to {iterations[-1]})")

sample_indices = [0, len(iterations)//4, len(iterations)//2, len(iterations)*3//4, len(iterations)-1]
print("\n" + "="*80)
print(f"{'Iteration':<12} | {'Mean Reward':<12} | {'Ep Length (steps)':<18} | {'TimeOut %':<12} | {'Value Loss':<12}")
print("="*80)
for idx in sample_indices:
    it = iterations[idx]
    r = rewards[idx]
    l = ep_lengths[idx]
    t = timeouts[idx] * 100
    v = val_losses[idx]
    print(f"{it:<12} | {r:<12.2f} | {l:<18.2f} | {t:<11.1f}% | {v:<12.1f}")
print("="*80)
