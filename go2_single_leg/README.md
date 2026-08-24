# Unitree Go2 1本足ダイナミックけんけん (Single-Leg Hopping)

1脚のみで連続跳躍しながらバランスを維持して移動する強化学習タスクです。

## 実行方法

### 学習
```bash
conda activate isaaclab
cd /home/exhibition-spakona/Desktop/REINFORCEMENT/go2_single_leg
python train.py --num_envs 4096 --max_iterations 1500
```

### 動画生成
```bash
python record_video.py --video_length 300
totem logs/go2_single_leg.mp4 &
```

### GUI再生
```bash
python play.py
```
