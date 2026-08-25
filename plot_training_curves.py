import re
import os
import matplotlib.pyplot as plt
import numpy as np

# ログファイル一覧（時系列順）
log_files = [
    "/home/exhibition-spakona/.gemini/antigravity-cli/brain/5911a837-9092-4123-82e7-95f98b9a8527/.system_generated/tasks/task-616.log",
    "/home/exhibition-spakona/.gemini/antigravity-cli/brain/5911a837-9092-4123-82e7-95f98b9a8527/.system_generated/tasks/task-658.log",
    "/home/exhibition-spakona/.gemini/antigravity-cli/brain/5911a837-9092-4123-82e7-95f98b9a8527/.system_generated/tasks/task-772.log",
    "/home/exhibition-spakona/.gemini/antigravity-cli/brain/5911a837-9092-4123-82e7-95f98b9a8527/.system_generated/tasks/task-826.log",
]

data = {}

for log_path in log_files:
    if not os.path.exists(log_path): continue
    with open(log_path, 'r') as f:
        content = f.read()
    
    blocks = content.split("Learning iteration ")
    for b in blocks[1:]:
        lines = b.split('\n')
        m_it = re.match(r'(\d+)/\d+', lines[0])
        if not m_it: continue
        it_num = int(m_it.group(1))
        
        m_rew = re.search(r'Mean reward:\s+([\d\.\-]+)', b)
        m_len = re.search(r'Mean episode length:\s+([\d\.\-]+)', b)
        m_val = re.search(r'Mean value loss:\s+([\d\.\-]+)', b)
        m_sur = re.search(r'Mean surrogate loss:\s+([\d\.\-]+)', b)
        m_time = re.search(r'Episode_Termination/time_out:\s+([\d\.\-]+)', b)
        m_launch = re.search(r'Episode_Reward/explosive_launch:\s+([\d\.\-]+)', b)
        m_flight = re.search(r'Episode_Reward/flight_clearance:\s+([\d\.\-]+)', b)
        
        if m_rew and m_len:
            rew = float(m_rew.group(1))
            ep_len = float(m_len.group(1))
            val_loss = float(m_val.group(1)) if m_val else 0.0
            sur_loss = float(m_sur.group(1)) if m_sur else 0.0
            time_out = float(m_time.group(1)) * 100 if m_time else 0.0
            launch = float(m_launch.group(1)) if m_launch else 0.0
            flight = float(m_flight.group(1)) if m_flight else 0.0
            
            data[it_num] = {
                'reward': rew,
                'ep_len': ep_len,
                'val_loss': val_loss,
                'sur_loss': sur_loss,
                'time_out': time_out,
                'launch': launch,
                'flight': flight,
            }

sorted_iters = sorted(data.keys())
print(f"Total Unique Iterations Parsed: {len(sorted_iters)} (from {sorted_iters[0]} to {sorted_iters[-1]})")

iters = np.array(sorted_iters)
rewards = np.array([data[i]['reward'] for i in sorted_iters])
ep_lens = np.array([data[i]['ep_len'] for i in sorted_iters])
val_losses = np.array([data[i]['val_loss'] for i in sorted_iters])
sur_losses = np.array([data[i]['sur_loss'] for i in sorted_iters])
timeouts = np.array([data[i]['time_out'] for i in sorted_iters])
launches = np.array([data[i]['launch'] for i in sorted_iters])
flights = np.array([data[i]['flight'] for i in sorted_iters])

# 移動平均平滑化
def smooth(y, box_pts=25):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=200)

# 1. 報酬の推移
axs[0, 0].plot(iters, rewards, color='#1f77b4', alpha=0.25, label='Raw Mean Reward')
axs[0, 0].plot(iters, smooth(rewards, 35), color='#1f77b4', linewidth=2.5, label='Smoothed Reward (35-pt MA)')
axs[0, 0].axvline(1000, color='gray', linestyle='--', alpha=0.7, label='Stage 1 (1k)')
axs[0, 0].axvline(2500, color='orange', linestyle='--', alpha=0.7, label='Stage 2 (2.5k)')
axs[0, 0].axvline(5000, color='green', linestyle='--', alpha=0.7, label='Stage 3 (5k)')
axs[0, 0].set_title('1. Mean Reward Trend (0 to 7400 Iterations)', fontsize=13, fontweight='bold')
axs[0, 0].set_xlabel('Learning Iterations', fontsize=11)
axs[0, 0].set_ylabel('Reward Score', fontsize=11)
axs[0, 0].legend(loc='upper left', frameon=True)
axs[0, 0].set_ylim(-10, 250)

# 2. 5秒完走率 & 平均エピソード長
ax2 = axs[0, 1].twinx()
p1 = axs[0, 1].plot(iters, smooth(timeouts, 35), color='#d62728', linewidth=2.5, label='5.0s Completion Rate (%)')
p2 = ax2.plot(iters, smooth(ep_lens, 35), color='#2ca02c', linewidth=2.0, linestyle='-.', label='Mean Episode Length (steps)')
axs[0, 1].set_title('2. 5.0s TimeOut Completion Rate & Episode Length', fontsize=13, fontweight='bold')
axs[0, 1].set_xlabel('Learning Iterations', fontsize=11)
axs[0, 1].set_ylabel('5.0s Completion Rate (%)', color='#d62728', fontsize=11)
ax2.set_ylabel('Mean Episode Length (steps)', color='#2ca02c', fontsize=11)
lines = p1 + p2
labels = [l.get_label() for l in lines]
axs[0, 1].legend(lines, labels, loc='upper left', frameon=True)

# 3. 打ち上げ推力 & 空中クリアランス報酬
axs[1, 0].plot(iters, smooth(launches, 35), color='#9467bd', linewidth=2.2, label='Explosive Launch Reward (Thrust)')
axs[1, 0].plot(iters, smooth(flights, 35), color='#8c564b', linewidth=2.2, label='Flight Air Clearance Reward')
axs[1, 0].set_title('3. Jump Dynamics Components (Launch vs Flight)', fontsize=13, fontweight='bold')
axs[1, 0].set_xlabel('Learning Iterations', fontsize=11)
axs[1, 0].set_ylabel('Reward Weight Metric', fontsize=11)
axs[1, 0].legend(loc='upper left', frameon=True)

# 4. 損失関数の推移 (Surrogate Loss & Value Loss)
valid_mask = (val_losses > 0) & (val_losses < 50000)
axs[1, 1].plot(iters[valid_mask], np.log10(val_losses[valid_mask] + 1e-3), color='#e377c2', alpha=0.3, label='Log10(Value Loss)')
axs[1, 1].plot(iters[valid_mask], smooth(np.log10(val_losses[valid_mask] + 1e-3), 35), color='#e377c2', linewidth=2.2, label='Smoothed Log10(Value Loss)')
axs[1, 1].set_title('4. Critic Value Loss Convergence (Log Scale)', fontsize=13, fontweight='bold')
axs[1, 1].set_xlabel('Learning Iterations', fontsize=11)
axs[1, 1].set_ylabel('Log10(Value Loss)', fontsize=11)
axs[1, 1].legend(loc='upper right', frameon=True)

plt.tight_layout()
out_plot_path = "/home/exhibition-spakona/Desktop/REINFORCEMENT/training_curves_7400_master.png"
plt.savefig(out_plot_path, dpi=250)
print(f"Saved full training curves plot to {out_plot_path}")
