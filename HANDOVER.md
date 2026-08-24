# 開発・引き継ぎ書

環境構築手順、設定ファイルの場所、試行錯誤の知見をまとめた引き継ぎドキュメントです。

---

## 開発方針

* 環境設定や報酬関数の意味を理解しながらチューニングを進める
* TensorBoard や GUI 再生で動作を観察し、仮説検証のサイクルを回す
* 3脚けんけんから始め、同側2本足歩行、1本足けんけんへと段階的に進める

---

## 環境セットアップ手順

### 前提環境
* OS: Ubuntu 24.04 LTS
* GPU: NVIDIA GeForce RTX 4060

### セットアップコマンド
```bash
conda create -n isaaclab python=3.12 -y
conda activate isaaclab

cd /home/exhibition-spakona/Desktop/REINFORCEMENT/IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
pip install -e source/isaaclab_assets
pip install rsl-rl-lib gymnasium matplotlib
```

---

## フォルダ構成

* `README.md`: GitHub成果物メインページ
* `HANDOVER.md`: 本ドキュメント
* `IsaacLab/`: コアフレームワーク
* `go2_right_side/`: 右側2脚歩行パッケージ（成功モデル・スクリプト一式）
* `go2_hopping/`: 3脚けんけん歩行パッケージ（成功モデル・スクリプト一式）
* `go2_single_leg/`: 1本足ダイナミックけんけん実験パッケージ
* `go2_bipedal/`: 後足2本立ち歩行実験パッケージ

---

## 試行錯誤の知見

### 1. 空中傾きスポーンの回避と遷移型カリキュラム
空中で極端に傾けた姿勢でスポーンさせると着地衝撃で即座に転倒します。
自然な4足接地で着地を受け止めてから、目標の脚をリフトして歩行モードへ移行させるアプローチが極めて安定します。

### 2. 匍匐前進（引きずり）の完全封鎖
足先のみをペナルティにすると、膝や肘、胸を床に擦りながら進む Reward Hacking が発生します。
* 全リンク（頭部・肘・太もも・胴体）の接触判定（`ContactSensor`）
* 最低高度判定（`root_height_below_minimum`）
を併用することで完全な空中保持歩行を獲得できます。

### 3. 静止して足を上げるだけの局所解への対策
足上げ報酬だけで高得点が入る状態だと、ロボットが前進せずにその場で固まる現象が発生します。
* 速度追従の計算をロボット座標系に設定
* 支持脚の滞空時間報酬を強化
* 静止ペナルティ（`stand_still_penalty`）の導入

---

## TensorBoard で確認する主な指標
* `Episode_Reward/track_lin_vel_xy_exp`: 目標速度への追従度
* `Metrics/success_rate`: 追従成功率
* `Episode_Reward/disabled_legs_height`: 浮かせ脚の高さ報酬
* `Episode_Termination/time_out`: エピソード完走率
