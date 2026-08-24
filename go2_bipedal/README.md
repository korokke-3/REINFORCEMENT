# Unitree Go2 後足2本立ち歩行 (Bipedal Locomotion) 強化学習ガイド

このパッケージは、**Isaac Sim + Isaac Lab + RSL-RL (PPO)** を用いて、四足歩行ロボット **Unitree Go2** に「後足2本立ち歩行（前足2本を浮かせ、後足2本だけでバランスを取りながら前進・旋回する）」スキルを獲得させるための強化学習環境です。

---

## 📁 ディレクトリ構成

```bash
go2_bipedal/
├── envs/
│   ├── __init__.py                # Gym環境ID (Isaac-Velocity-Rough-Unitree-Go2-Bipedal-v0)
│   ├── go2_bipedal_env_cfg.py     # ★ 環境設定 (報酬重み・前足浮かせ・直立姿勢)
│   └── go2_bipedal_rewards.py     # ★ 2本足専用報酬計算 (前足ペナルティ、ピッチ角報酬など)
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
cd /home/exhibition-spakona/Desktop/REINFORCEMENT/go2_bipedal

# 4096並列・1500イテレーションで学習
python train.py --num_envs 4096 --max_iterations 1500
```

### 2. 学習ログの確認 (TensorBoard)
```bash
tensorboard --logdir /home/exhibition-spakona/Desktop/REINFORCEMENT/go2_bipedal/logs/rsl_rl --port 6007
# ブラウザで http://localhost:6007 を開く
```

### 3. 動画の作成 (Record Video)
```bash
python record_video.py --video_length 300
totem logs/go2_bipedal.mp4 &
```

### 4. 3D GUIでの再生 (Play)
```bash
python play.py
```
