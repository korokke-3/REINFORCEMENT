# 開発・引き継ぎ書

環境構築手順、設定ファイルの場所、報酬設計の知見をまとめたドキュメントです。

---

## 開発方針

* 3脚けんけん歩行、右側2脚走行、単脚連続跳躍へと段階的に難易度を上げて学習を進める
* TensorBoard および GUI 再生で動作を観察し、報酬とペナルティのバランスを調整する

---

## 環境セットアップ

### 前提環境
* OS: Ubuntu 22.04 / 24.04 LTS
* GPU: NVIDIA GeForce RTX 3060以上 (CUDA 12対応)

### コマンド
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

## フォルダ構成

* `README.md`: メイン解説ドキュメント
* `HANDOVER.md`: 本ドキュメント
* `models/`: 学習済みベストモデル重み
* `assets/`: デモGIF・画像・動画
* `go2_single_leg/`: 単脚連続跳躍 パッケージ
* `go2_hopping/`: 3脚けんけん歩行 パッケージ
* `go2_right_side/`: 右側2脚走行 パッケージ
* `eval_single_leg_master.py`: 単脚跳躍モデル評価スクリプト

---

## 報酬設計の知見

### 1. 着地衝撃の緩和と遷移型カリキュラム
極端に傾いた姿勢からスポーンすると着地衝撃で転倒しやすいため、4脚接地姿勢で安定させてから目標脚のリフトへ移行する設計が有効です。

### 2. 接地判定による這い歩き防止
足先のみを接地ペナルティにすると、膝や胴体を床に擦りながら進む不正解が発生します。全リンクの接触判定（`ContactSensor`）および最低高度判定（`root_height_below_minimum`）を併用して空中姿勢を保持させます。

### 3. 静止足上げ局所解の回避
足上げ報酬単体では「その場で静止して足を上げ続ける」状態になるため、支持脚の滞空時間報酬（`hopping_feet_air_time`）と速度追従報酬を組み合わせて推進力を与えます。
