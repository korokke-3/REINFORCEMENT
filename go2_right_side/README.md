# Unitree Go2 右側2脚走行 (Right-Side Bipedal Running) 強化学習ガイド

このパッケージは、**Isaac Sim + Isaac Lab + RSL-RL (PPO)** を用いて、四足歩行ロボット **Unitree Go2** に「同側2脚歩行（左側2脚を完全に浮かせ、右前脚と右後脚のみでバランスを取りながら走行する）」スキルを獲得させるための強化学習環境です。

---

## 📁 ディレクトリ構成

```bash
go2_right_side/
├── envs/
│   ├── __init__.py                # Gym環境ID (Isaac-Velocity-Rough-Unitree-Go2-RightSide-v0)
│   ├── go2_right_side_env_cfg.py  # ★ 環境設定 (報酬重み・左足浮かせ・ロール角姿勢)
│   └── go2_right_side_rewards.py  # ★ 右側2脚専用報酬計算 (左足ペナルティ、横方向バランス報酬)
├── agents/
│   └── rsl_rl_ppo_cfg.py          # PPOハイパーパラメータ設定
├── train.py                       # 学習実行スクリプト (GPU 4096並列・Headless)
├── play.py                        # GUI 3Dビューアー再生スクリプト
├── record_video.py                # MP4動画生成スクリプト
├── analyze_trajectory.py          # 軌跡・速度解析スクリプト
└── logs/                          # ログ・チェックポイント・動画保存先
```

---

## 🚀 実行コマンド

### 1. 学習の開始 (Train)
```bash
conda activate isaaclab
cd /home/exhibition-spakona/Desktop/REINFORCEMENT/go2_right_side

# 4096並列・1500イテレーションで学習
python train.py --num_envs 4096 --max_iterations 1500
```

### 2. 学習ログの確認 (TensorBoard)
```bash
tensorboard --logdir logs/rsl_rl --port 6008
# ブラウザで http://localhost:6008 を開く
```

### 3. 動画の作成 (Record Video)
```bash
python record_video.py --video_length 300
totem logs/go2_right_side.mp4 &
```

### 4. 3D GUIでの再生 (Play)
```bash
python play.py
```
