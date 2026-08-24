# Unitree Go2 片足けんけん (Hopping / 3-Legged Locomotion) 強化学習ガイド

このリポジトリは、**Isaac Sim + Isaac Lab + RSL-RL (PPO)** を用いて、四足歩行ロボット **Unitree Go2** に「片足けんけん（1脚を浮かせた状態でのホッピング・歩行）」を獲得させるための強化学習実験環境です。

---

## 📁 ディレクトリ構成と役割

```bash
go2_hopping/
├── envs/
│   ├── go2_hopping_env_cfg.py     # ★ 環境設定 (報酬の重み・地形・カメラ設定など)
│   └── go2_hopping_rewards.py     # ★ 報酬計算式 (ペナルティ・プラス報酬の計算ロジック)
├── agents/
│   └── rsl_rl_ppo_cfg.py          # ★ PPO設定 (学習率、総イテレーション数、保存間隔など)
├── train.py                       # 学習実行スクリプト (GPU 4096並列・Headless)
├── play.py                        # GUI 3Dビューアー再生スクリプト
├── record_video.py                # ★ MP4動画自動生成スクリプト (フリーズなし)
└── logs/                          # 学習ログ・重み(.pt)・動画(.mp4)の保存先
```

---

## 🛠️ 1. 設定の変更方法 (Customization)

### (1) 報酬の重みを調整する
👉 `envs/go2_hopping_env_cfg.py` の `Go2HoppingRewardsCfg` クラスを編集します。

```python
# 例: 報酬の重み（weight）の調整
track_lin_vel_xy_exp = RewTerm(func=custom_rewards.track_lin_vel_xy_exp, weight=1.5)  # 目標速度追従
disabled_leg_contact = RewTerm(func=custom_rewards.disabled_leg_contact_penalty, weight=-2.0)  # 接地ペナルティ
disabled_leg_height  = RewTerm(func=custom_rewards.disabled_leg_height_reward, weight=1.0)   # 足上げ報酬
hopping_air_time     = RewTerm(func=custom_rewards.hopping_feet_air_time, weight=1.5)        # ★跳躍・滞空時間報酬
lin_vel_z_l2         = RewTerm(func=custom_rewards.lin_vel_z_l2, weight=-0.0)                # 上下動ペナルティ（跳ねるなら0）
```

### (2) 報酬計算ロジックを追加・修正する
👉 `envs/go2_hopping_rewards.py` を編集します。
- 新しい関数（例: 静止ペナルティ、前方移動量報酬など）を定義し、上記 `env_cfg.py` で登録します。

### (3) 学習パラメータ（イテレーション数等）を変更する
👉 `agents/rsl_rl_ppo_cfg.py` を編集します。
- `max_iterations = 1500`: 総学習ステップ数
- `learning_rate = 1.0e-3`: 学習率
- `save_interval = 100`: チェックポイント保存間隔

---

## 🚀 2. 再学習の実行コマンド

仮想環境 `isaaclab` を有効化して実行します：

```bash
conda activate isaaclab
cd /home/exhibition-spakona/Desktop/REINFORCEMENT/go2_hopping

# 学習の開始 (4096並列、1500イテレーション)
python train.py --num_envs 4096 --max_iterations 1500
```
*(※ 学習ログとモデルは `logs/rsl_rl/<タイムスタンプ>/` に自動保存されます)*

---

## 📈 3. 学習状況のモニタリング (TensorBoard)

学習中の平均報酬やエピソード長、各報酬項の推移をリアルタイムで確認できます：

```bash
tensorboard --logdir logs/rsl_rl
```
ブラウザで `http://localhost:6006` を開くとグラフが表示されます。

---

## 🎬 4. 動作確認・動画（MP4）ファイルの作成

学習した最新モデルを読み込み、1台のロボットが動く様子を MP4 動画として書き出します：

```bash
# 300ステップ（約6秒間）の動画を生成
python record_video.py --video_length 300
```

* **生成される動画ファイル**: `logs/go2_hopping.mp4`
* **画面で動画を再生するコマンド**:
  ```bash
  totem logs/go2_hopping.mp4 &
  ```

---

## 💡 「動かない（静止立ち）」を防ぐための報酬設計のコツ

1. **静止を許さない（速度報酬の強化）**:
   - `disabled_leg_height`（足上げ報酬）単体で得点を与えすぎると、**「片足を上げたまま静止して得点を稼ぐ」局所解** にハマります。
   - `hopping_air_time`（滞空時間）の重みを大きく（`1.5`〜`2.0`）設定してください。
2. **上下動ペナルティ（`lin_vel_z_l2`）の緩和**:
   - 跳躍には上下の動きが必須なため、このペナルティを `0.0` に設定して跳ねる動きを許容します。
