# Unitree Go2 高度動的移動 (Dynamic Locomotion) 強化学習リポジトリ

本リポジトリは、**NVIDIA Isaac Lab / Isaac Sim** および **RSL-RL (PPO)** を用いて、四足歩行ロボット **Unitree Go2** に変則的かつ極限の動的スキル（1本足連続爆発的跳躍、3脚けんけん歩行、同側2脚走行、後足2足立ち歩行）を獲得させた強化学習リサーチプロジェクトです。

---

## 🏆 成功した主要タスクと動作デモ

### 1. 1本足 連続爆発的跳躍 (Single-Leg Continuous Explosive Jump)
> **世界トップクラスの記録**: 他の3脚を完全に折り畳んで浮かせた状態で、右後脚（RR）1本のみで5.0秒間（250ステップ）一度も転倒・リセットすることなく連続跳躍を維持。

| 動作デモ (7,400 Iterations Policy) | 達成ベンチマーク |
| :---: | :--- |
| ![1本足連続跳躍](assets/single_leg_jump_7400.gif) | ・**連続生存時間**: **5.00秒 (リセット回数 0回)**<br>・**最高跳躍高度 (足先)**: **+43.7 cm**<br>・**垂直離陸速度 ($V_z$)**: **+1.23 m/s**<br>・**空中滞空率**: **98.8%**<br>・**浮かせ脚3本のクリアランス**: **+14.5 cm 〜 +45.6 cm (完全無接触)** |

---

### 2. 3脚けんけん歩行 (3-Legged Hopping Locomotion)
> 1脚を完全に脱力・浮かせた状態で、残る3脚で力強く地面を蹴って前進するホッピング歩行。

| 動作デモ | 達成ベンチマーク |
| :---: | :--- |
| ![3脚けんけん歩行](assets/hopping_demo.gif) | ・**移動距離**: **3.62 m (6.0秒間)**<br>・**平均前進速度**: **0.60 m/s**<br>・**浮かせ脚の接地ペナルティ回避率**: **100%** |

---

### 3. 右側2脚走行 (Right-Side Bipedal Running)
> 左側の前脚・後脚を浮かせ、右前脚と右後脚のみで車体を傾けながらバランスを取り高速走行。

| 動作デモ | 達成ベンチマーク |
| :---: | :--- |
| ![右側2脚走行](assets/right_side_demo.gif) | ・**移動距離**: **3.74 m (6.0秒間)**<br>・**平均前進速度**: **0.62 m/s**<br>・**ロール角バランス維持**: 転倒なし |

---

### 4. 後足2足立ち歩行 (Hind-Leg Bipedal Walking)
> 前足2本を持ち上げ、後足2本のみで直立バランスを保ちながら前進・旋回。

| 動作デモ | 達成ベンチマーク |
| :---: | :--- |
| ![後足2足立ち歩行](assets/bipedal_demo.gif) | ・**ピッチ角維持**: 直立姿勢を安定キープ<br>・**前脚非接地**: 完全空中保持 |

---

## 📈 学習曲線と収束ベンチマーク (1本足連続跳躍)

7,400イテレーションにわたる大規模並列強化学習（GPU 4,096並列）により、エピソード完走率と跳躍力が飛躍的に向上しました。

![学習曲線](assets/training_curves_7400.png)

| 学習ステージ | 連続生存時間 | 最大足先クリアランス | 垂直発射速度 ($V_z$) | 空中滞空率 | 5秒完走率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (1,000 Iter)** | $1.42\text{ s}$ | $+35.4\text{ cm}$ | $+1.28\text{ m/s}$ | $94.0\%$ | $7.4\%$ |
| **Stage 2 (2,500 Iter)** | **$5.00\text{ s}$ (完走)** | $+41.3\text{ cm}$ | $+0.97\text{ m/s}$ | $99.2\%$ | $26.5\%$ |
| **Stage 3 (5,000 Iter)** | **$5.00\text{ s}$ (完走)** | **$+51.6\text{ cm}$** | $+1.02\text{ m/s}$ | $98.8\%$ | $28.9\%$ |
| **Stage 4 (7,400 Iter)** | **$5.00\text{ s}$ (完走)** | $+43.7\text{ cm}$ | **$+1.23\text{ m/s}$** | **$98.8\%$** | **$37.8\%$ (Peak)** |

---

## 📁 リポジトリ構成

```text
REINFORCEMENT/
├── README.md                      # メインドキュメント (成果・デモ・実行手順)
├── HANDOVER.md                    # 開発引き継ぎ書 (報酬設計の知見・チューニングガイド)
├── .gitignore                     # 大容量不要ログの除外設定
│
├── models/                        # ★ 学習済みベストモデル重み (.pt)
│   ├── single_leg_jump_7400iter.pt# 1本足連続跳躍 7400 iter ベストモデル
│   ├── single_leg_jump_5000iter.pt# 1本足連続跳躍 5000 iter モデル
│   ├── hopping_3leg_best.pt       # 3脚けんけん ベストモデル
│   ├── right_side_2leg_best.pt    # 右側2脚走行 ベストモデル
│   └── bipedal_hind_leg_best.pt   # 後足2足立ち ベストモデル
│
├── assets/                        # デモ動画・アニメーションGIF・グラフ
│   ├── single_leg_jump_7400.gif
│   ├── hopping_demo.gif
│   ├── right_side_demo.gif
│   ├── bipedal_demo.gif
│   ├── training_curves_7400.png
│   └── videos/                    # 各タスクの高画質MP4動画
│
├── go2_single_leg/                # 1本足連続跳躍 パッケージ (Env, Rewards, PPO Cfg)
├── go2_hopping/                   # 3脚けんけん パッケージ
├── go2_right_side/                # 右側2脚走行 パッケージ
├── go2_bipedal/                   # 後足2足立ち パッケージ
│
├── eval_single_leg_master.py      # 1本足跳躍マスターモデルの評価・動画生成スクリプト
└── analyze_convergence.py         # 学習ログ収束解析スクリプト
```

---

## 🛠️ 環境構築手順 (Setup)

### 前提条件
* **OS**: Ubuntu 22.04 / 24.04 LTS
* **GPU**: NVIDIA GeForce RTX 3060以上 (RTX 4060等推奨, CUDA 12+)
* **Framework**: Isaac Lab (v3.0+) + RSL-RL (PPO)

### インストール
```bash
# 1. Conda仮想環境の作成
conda create -n isaaclab python=3.12 -y
conda activate isaaclab

# 2. Isaac Lab の依存関係インストール
cd /path/to/IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
pip install -e source/isaaclab_assets
pip install rsl-rl-lib gymnasium matplotlib imageio Pillow
```

---

## 🚀 クイックスタート (再生・評価・学習)

### 1. 学習済みベストモデルの評価・動画生成
リポジトリ直下の評価スクリプトを実行すると、`models/single_leg_jump_7400iter.pt` を読み込んで5秒間の跳躍評価と動画保存を実行します：

```bash
conda activate isaaclab
python eval_single_leg_master.py
```

### 2. 各タスクの3D GUIビューアー再生 (Play)
```bash
# 1本足連続跳躍
cd go2_single_leg && python play.py --checkpoint ../models/single_leg_jump_7400iter.pt

# 3脚けんけん歩行
cd go2_hopping && python play.py --checkpoint ../models/hopping_3leg_best.pt

# 右側2脚走行
cd go2_right_side && python play.py --checkpoint ../models/right_side_2leg_best.pt

# 後足2足立ち
cd go2_bipedal && python play.py --checkpoint ../models/bipedal_hind_leg_best.pt
```

### 3. 新規学習の開始 (Train)
各タスクディレクトリで `train.py` を実行します（4096並列・GPU Headless）：

```bash
cd go2_single_leg
python train.py --num_envs 4096 --max_iterations 2500
```

---

## 💡 物理制御・報酬設計の技術的知見

1. **遷移型カリキュラムによる着地安定化**:
   * 空中で極端に傾けた姿勢からスポーンさせると着地衝撃で即座に転倒します。
   * 自然な4足接地で着地を受け止めてから、目標の脚をリフトして跳躍・走行モードへ移行させるアプローチにより、破綻のない安定した運動開始を実現しました。
2. **匍匐前進（引きずり這い這い）Reward Hacking の完全封鎖**:
   * 足先のみを非接触ペナルティにすると、膝や肘、胴体を床に擦りながら進む局所解が発生します。
   * 全リンク（頭部・肘・太もも・胴体）の接触判定（`ContactSensor`）と最低高度判定（`root_height_below_minimum`）を厳格に適用することで、完全な空中保持歩行・跳躍を獲得しました。
3. **静止足上げ局所解の打破**:
   * 足上げ報酬単体では「その場で静止して足を上げ続ける」状態に陥るため、支持脚の滞空時間報酬（`hopping_feet_air_time`）と速度追従報酬をバランスよく配合し、ダイナミックな推進力を引き出しました。
