# Unitree Go2 単脚連続跳躍 強化学習

Isaac Lab および RSL-RL を使用した、Unitree Go2 による単脚連続跳躍および変則移動制御の強化学習リポジトリです。

---

## 1. 単脚連続跳躍

右後脚（RR）1本のみで接地・跳躍を行い、他の3脚を浮かせた状態で連続跳躍を維持します。

### 学習イテレーション別の動作比較

| 1,000 Iterations | 2,500 Iterations |
| :---: | :---: |
| ![1000 iter](assets/single_leg_1000iter.gif) | ![2500 iter](assets/single_leg_2500iter.gif) |
| **初期段階**<br>接地姿勢の保持が中心。生存時間 約1.4秒。 | **完走達成段階**<br>5.0秒間のエピソード完走を達成。 |

| 5,000 Iterations | 7,400 Iterations |
| :---: | :---: |
| ![5000 iter](assets/single_leg_5000iter.gif) | ![7400 iter](assets/single_leg_7400iter.gif) |
| **高跳躍段階**<br>足先クリアランス最大 +51.6 cm を記録。 | **最終収束段階**<br>着地と離陸が連動し、完走率 37.8% を達成。 |

---

## 2. 定量的ベンチマーク

![学習曲線](assets/training_curves_7400.png)

| 学習ステージ | 連続生存時間 | 最大足先クリアランス | 垂直離陸速度 $V_z$ | 空中滞空率 | 5秒完走率 |
| :--- | :--- | :--- | :--- | :--- |
| 1,000 Iterations | 1.42 秒 | +35.4 cm | +1.28 m/s | 94.0% | 7.4% |
| 2,500 Iterations | 5.00 秒 | +41.3 cm | +0.97 m/s | 99.2% | 26.5% |
| 5,000 Iterations | 5.00 秒 | +51.6 cm | +1.02 m/s | 98.8% | 28.9% |
| 7,400 Iterations | 5.00 秒 | +43.7 cm | +1.23 m/s | 98.8% | 37.8% |

浮かせた3脚（FL, FR, RL）は高度 +14.5 cm 〜 +45.6 cm を維持し、床接触は発生しません。

---

## 3. その他の移動タスク

| 3脚けんけん歩行 | 右側2脚走行 |
| :---: | :---: |
| ![3脚けんけん歩行](assets/hopping_demo.gif) | ![右側2脚走行](assets/right_side_demo.gif) |
| 1脚を浮かせた状態でのホッピング前進（6秒間で3.62 m移動） | 左側2脚を浮かせた右側2脚での走行（6秒間で3.74 m移動） |

---

## 4. リポジトリ構成

```text
REINFORCEMENT/
├── README.md                      # メインドキュメント
├── HANDOVER.md                    # 開発引き継ぎ書
├── .gitignore
│
├── models/                        # 学習済みモデル
│   ├── single_leg_jump_7400iter.pt# 単脚跳躍 7400 iter モデル
│   ├── single_leg_jump_5000iter.pt# 単脚跳躍 5000 iter モデル
│   ├── hopping_3leg_best.pt       # 3脚けんけんモデル
│   └── right_side_2leg_best.pt    # 右側2脚走行モデル
│
├── assets/                        # デモGIF・画像・動画
│   ├── single_leg_1000iter.gif
│   ├── single_leg_2500iter.gif
│   ├── single_leg_5000iter.gif
│   ├── single_leg_7400iter.gif
│   ├── training_curves_7400.png
│   └── videos/
│
├── go2_single_leg/                # 単脚連続跳躍 環境・報酬・学習コード
├── go2_hopping/                   # 3脚けんけん 環境・報酬・学習コード
├── go2_right_side/                # 右側2脚走行 環境・報酬・学習コード
│
├── eval_single_leg_master.py      # 単脚跳躍モデル評価スクリプト
└── analyze_convergence.py         # 学習ログ解析スクリプト
```

---

## 5. 環境セットアップ

```bash
conda create -n isaaclab python=3.12 -y
conda activate isaaclab

cd /path/to/IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
pip install -e source/isaaclab_assets
pip install rsl-rl-lib gymnasium matplotlib imageio Pillow
```

---

## 6. 実行方法

### モデル評価
7,400イテレーションの単脚跳躍モデルを評価し、5秒間の動画を保存します。

```bash
conda activate isaaclab
python eval_single_leg_master.py
```

### シミュレーション再生
```bash
# 単脚連続跳躍
cd go2_single_leg && python play.py --checkpoint ../models/single_leg_jump_7400iter.pt

# 3脚けんけん歩行
cd go2_hopping && python play.py --checkpoint ../models/hopping_3leg_best.pt

# 右側2脚走行
cd go2_right_side && python play.py --checkpoint ../models/right_side_2leg_best.pt
```

### 学習実行
```bash
cd go2_single_leg
python train.py --num_envs 4096 --max_iterations 2500
```
